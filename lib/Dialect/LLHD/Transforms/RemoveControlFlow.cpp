//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-remove-control-flow"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_REMOVECONTROLFLOWPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// A helper struct that tracks a boolean condition as either a constant false,
/// constant true, or an SSA value.
struct Condition {
  Condition() {}
  Condition(Value value) : pair(value, 0) {
    if (value) {
      if (matchPattern(value, m_One()))
        *this = Condition(true);
      if (matchPattern(value, m_Zero()))
        *this = Condition(false);
    }
  }
  Condition(bool konst) : pair(nullptr, konst ? 1 : 2) {}

  explicit operator bool() const {
    return pair.getPointer() != nullptr || pair.getInt() != 0;
  }

  bool isTrue() const { return !pair.getPointer() && pair.getInt() == 1; }
  bool isFalse() const { return !pair.getPointer() && pair.getInt() == 2; }
  Value getValue() const { return pair.getPointer(); }

  /// Turn this condition into an SSA value, creating an `hw.constant` if the
  /// condition is a constant.
  Value materialize(OpBuilder &builder, Location loc) const {
    if (isTrue())
      return hw::ConstantOp::create(builder, loc, APInt(1, 1));
    if (isFalse())
      return hw::ConstantOp::create(builder, loc, APInt(1, 0));
    return pair.getPointer();
  }

  Condition orWith(Condition other, OpBuilder &builder) const {
    if (isTrue() || other.isTrue())
      return true;
    if (isFalse())
      return other;
    if (other.isFalse())
      return *this;
    return builder.createOrFold<comb::OrOp>(getValue().getLoc(), getValue(),
                                            other.getValue());
  }

  Condition andWith(Condition other, OpBuilder &builder) const {
    if (isFalse() || other.isFalse())
      return false;
    if (isTrue())
      return other;
    if (other.isTrue())
      return *this;
    return builder.createOrFold<comb::AndOp>(getValue().getLoc(), getValue(),
                                             other.getValue());
  }

  Condition inverted(OpBuilder &builder) const {
    if (isTrue())
      return false;
    if (isFalse())
      return true;
    return comb::createOrFoldNot(builder, getValue().getLoc(), getValue());
  }

  bool operator==(const Condition &other) const { return pair == other.pair; }

  /// `sel ? onTrue : onFalse` over conditions, folding every case in which
  /// one arm is constant or both arms agree, so that a selector whose two
  /// arms are equal drops out entirely.
  static Condition select(Condition sel, Condition onTrue, Condition onFalse,
                          OpBuilder &builder) {
    if (sel.isTrue())
      return onTrue;
    if (sel.isFalse())
      return onFalse;
    if (onTrue == onFalse)
      return onTrue;
    if (onTrue.isTrue() && onFalse.isFalse())
      return sel;
    if (onTrue.isFalse() && onFalse.isTrue())
      return sel.inverted(builder);
    if (onTrue.isTrue())
      return sel.orWith(onFalse, builder);
    if (onTrue.isFalse())
      return sel.inverted(builder).andWith(onFalse, builder);
    if (onFalse.isTrue())
      return sel.inverted(builder).orWith(onTrue, builder);
    if (onFalse.isFalse())
      return sel.andWith(onTrue, builder);
    auto loc = sel.getValue().getLoc();
    return builder.createOrFold<comb::MuxOp>(loc, sel.getValue(),
                                             onTrue.getValue(),
                                             onFalse.getValue());
  }

private:
  llvm::PointerIntPair<Value, 2> pair;
};
} // namespace

//===----------------------------------------------------------------------===//
// Structured Evaluation of a Join
//===----------------------------------------------------------------------===//

/// Both quantities this pass needs at a block -- the values its block
/// arguments take, and the condition under which control reaches it -- are
/// functions of the branch decisions taken between a dominator of the block
/// and the block itself. There are two ways to build them.
///
/// The flat way (what this pass used to do): enumerate the incoming EDGES of
/// the join, compute each edge's path predicate as an OR over all paths from
/// the dominator, and chain per-edge muxes: mux(p1, v1, mux(p2, v2, ...)).
/// Every path predicate mentions every branch condition on the path, so a
/// value that is the SAME along several edges is rebuilt as a disjunction of
/// the predicates that select it. For `req = 1; if (g) { if (r) a=..; else
/// b=..; }` the join after the inner `if` sees `req` arrive along three
/// edges with the same value, and the result is
/// `v&~g | v&g&r | v&g&~r`: a tautology in g and r that no local
/// canonicalization folds (it needs the knowledge that the three predicates
/// partition the space). `req` then structurally depends on g even though
/// it functionally does not, and when g is computed from req elsewhere the
/// design acquires a combinational cycle the RTL never had (CVA6's
/// store_buffer data_req/data_gnt, icache fetch_req, commit/csr; ibex's
/// addr_incr_req_o vs lsu_req_i). Yosys, which does structured
/// if-conversion in `proc_mux`, reports no such loop.
///
/// The structured way (this): walk the CFG forward from the dominator as
/// the nested if/else it was lowered from. A conditional branch is ONE mux on
/// its own condition whose arms are the values reaching the join through its
/// two successors; an unconditional branch passes the value of its successor
/// through; an edge into the join contributes the branch operands. Nested
/// control flow becomes nested muxes, mux(g, mux(r, v, v), v), and the
/// equal-arm fold that comb.mux already performs collapses each level whose
/// arms agree, so the value above depends on nothing but v. A successor that
/// cannot reach the join at all (control leaves the sub-region) contributes
/// no arm: the branch is not a decision about the join's arguments, and the
/// other arm is taken directly. This is also what makes the result minimal
/// for the common case of a single predecessor whose branch either enters
/// the block or skips it. Every block on a path from the dominator to the
/// join is dominated by the dominator and the region is acyclic (checked up
/// front), so the recursion is a DAG evaluation; results are memoised per
/// block so shared sub-paths yield shared muxes.
namespace {
template <typename T>
class StructuredJoin {
public:
  /// `leaf(operands)` maps an edge into `target` to the quantity being
  /// evaluated; `absent()` is the quantity (as a `std::optional<T>`) for a
  /// successor that does not reach `target`, where `std::nullopt` means the
  /// arm is dropped and the branch passes its other arm through;
  /// `select(cond, t, f)` combines the two arms of a conditional branch.
  StructuredJoin(DominanceInfo &domInfo, Block *domBlock, Block *target)
      : target(target) {
    // Collect the blocks lying on some path from `domBlock` to `target`
    // (both included) with a backward walk that stops at `domBlock`.
    SmallVector<Block *> worklist{target};
    between.insert(target);
    while (!worklist.empty()) {
      auto *block = worklist.pop_back_val();
      if (block == domBlock)
        continue;
      for (auto *pred : block->getPredecessors())
        if (domInfo.isReachableFromEntry(pred) && between.insert(pred).second)
          worklist.push_back(pred);
    }
  }

  template <typename Leaf, typename Absent, typename Select>
  T evaluate(Block *from, Leaf leaf, Absent absent, Select select) {
    return evalBlock(from, leaf, absent, select);
  }

private:
  template <typename Leaf, typename Absent, typename Select>
  std::optional<T> evalEdge(Block *succ, ValueRange operands, Leaf &leaf,
                            Absent &absent, Select &select) {
    if (succ == target)
      return leaf(operands);
    if (!between.contains(succ))
      return absent();
    return evalBlock(succ, leaf, absent, select);
  }

  template <typename Leaf, typename Absent, typename Select>
  T evalBlock(Block *block, Leaf &leaf, Absent &absent, Select &select) {
    assert(block != target && between.contains(block));
    if (auto it = memo.find(block); it != memo.end())
      return it->second;

    T result;
    auto *terminator = block->getTerminator();
    if (auto brOp = dyn_cast<cf::BranchOp>(terminator)) {
      // The only successor of a block on a path to the target is itself on
      // that path.
      auto value = evalEdge(brOp.getDest(), brOp.getDestOperands(), leaf,
                            absent, select);
      assert(value && "successor of a block between dominator and target "
                      "must reach the target");
      result = std::move(*value);
    } else {
      auto condBrOp = cast<cf::CondBranchOp>(terminator);
      auto onTrue = evalEdge(condBrOp.getTrueDest(),
                             condBrOp.getTrueDestOperands(), leaf, absent,
                             select);
      auto onFalse = evalEdge(condBrOp.getFalseDest(),
                              condBrOp.getFalseDestOperands(), leaf, absent,
                              select);
      assert((onTrue || onFalse) && "block between dominator and target must "
                                    "reach the target");
      if (onTrue && onFalse)
        result = select(condBrOp.getCondition(), *onTrue, *onFalse);
      else if (onTrue)
        result = std::move(*onTrue);
      else
        result = std::move(*onFalse);
    }
    memo[block] = result;
    return result;
  }

  Block *target;
  SmallPtrSet<Block *, 8> between;
  DenseMap<Block *, T> memo;
};
} // namespace

/// Lower the block arguments of `target` to values: the branch operands
/// arriving along each edge, combined by the branch structure between
/// `domBlock` (which dominates `target`) and `target`.
static SmallVector<Value> lowerJoinArguments(OpBuilder &builder,
                                             DominanceInfo &domInfo,
                                             Block *domBlock, Block *target) {
  StructuredJoin<SmallVector<Value>> join(domInfo, domBlock, target);
  return join.evaluate(
      domBlock,
      [](ValueRange operands) { return SmallVector<Value>(operands); },
      // A successor that never reaches the join contributes no arm: the
      // arguments' values are irrelevant when control goes that way.
      []() -> std::optional<SmallVector<Value>> { return std::nullopt; },
      [&](Value cond, const SmallVector<Value> &onTrue,
          const SmallVector<Value> &onFalse) {
        SmallVector<Value> result;
        result.reserve(onTrue.size());
        Condition sel(cond);
        for (auto [t, f] : llvm::zip(onTrue, onFalse)) {
          if (t == f || sel.isTrue())
            result.push_back(t);
          else if (sel.isFalse())
            result.push_back(f);
          else
            result.push_back(
                builder.createOrFold<comb::MuxOp>(t.getLoc(), cond, t, f));
        }
        return result;
      });
}

/// Compute the condition under which control flowing out of `domBlock`
/// (which dominates `target`) reaches `target`, structured by the branches
/// in between so that a branch both of whose successors reach `target`
/// contributes nothing.
static Condition getReachCondition(OpBuilder &builder, DominanceInfo &domInfo,
                                   Block *domBlock, Block *target) {
  StructuredJoin<Condition> join(domInfo, domBlock, target);
  return join.evaluate(
      domBlock, [](ValueRange) { return Condition(true); },
      // A successor that never reaches the target is a way of NOT reaching
      // it: that arm is false, not dropped.
      []() -> std::optional<Condition> { return Condition(false); },
      [&](Value cond, Condition onTrue, Condition onFalse) {
        return Condition::select(Condition(cond), onTrue, onFalse, builder);
      });
}

//===----------------------------------------------------------------------===//
// Ordered Drive Merging
//===----------------------------------------------------------------------===//

/// After control flow removal, a combinational body is a single block of
/// drives in PROGRAM ORDER, with path conditions folded into enables. RTL
/// like CVA6's return-address stack elaborates to a default whole-signal
/// drive followed by conditional overrides of individual fields:
///
///   llhd.drv %stack_d, %stack_q after %t
///   llhd.drv (sig.struct_extract (sig.array_get %stack_d[0]) "ra"),
///            %data_i after %t if %push
///
/// Nothing downstream can consume this: CombineDrives groups drives by
/// {delay, enable} and never merges across groups, and Sig2Reg refuses
/// overlapping intervals outright. But within one sequential block the
/// resolution is fully defined -- later drives win -- so this folds each
/// root signal's ordered drives into ONE whole-signal drive whose value is
/// built with struct_inject/array_inject and muxes.
///
/// Array slices are updated element-wise with array_get/array_inject rather
/// than hw.array_concat, sidestepping any question of element order
/// conventions in concat.

/// Walk a drive target back to its root signal, collecting the projection
/// ops along the way (leaf-to-root order). Returns the root ref value.
static Value traceToRoot(Value ref, SmallVectorImpl<Operation *> &path) {
  while (auto *defOp = ref.getDefiningOp()) {
    if (isa<SigStructExtractOp, SigArrayGetOp, SigArraySliceOp, SigExtractOp>(
            defOp)) {
      path.push_back(defOp);
      ref = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return ref;
}

/// Rebuild `whole` (a value of the root signal's type) with the slice
/// addressed by `path` (leaf-to-root order, so iterate from the back)
/// replaced by `value`. Returns null if a projection has a shape we do not
/// handle (dynamic indices).
static Value rebuildInject(OpBuilder &builder, Location loc, Value whole,
                           ArrayRef<Operation *> path, Value value) {
  if (path.empty())
    return value;
  Operation *step = path.back();
  auto rest = path.drop_back();

  if (auto op = dyn_cast<SigStructExtractOp>(step)) {
    auto field = op.getFieldAttr();
    Value child = hw::StructExtractOp::create(builder, loc, whole, field);
    Value updated = rebuildInject(builder, loc, child, rest, value);
    if (!updated)
      return {};
    return hw::StructInjectOp::create(builder, loc, whole, field, updated);
  }
  if (auto op = dyn_cast<SigArrayGetOp>(step)) {
    // Only constant indices; a dynamic index would need a full demux.
    if (!op.getIndex().getDefiningOp<hw::ConstantOp>())
      return {};
    Value child = hw::ArrayGetOp::create(builder, loc, whole, op.getIndex());
    Value updated = rebuildInject(builder, loc, child, rest, value);
    if (!updated)
      return {};
    return hw::ArrayInjectOp::create(builder, loc, whole, op.getIndex(),
                                     updated);
  }
  if (auto op = dyn_cast<SigArraySliceOp>(step)) {
    auto lowConst = op.getLowIndex().getDefiningOp<hw::ConstantOp>();
    if (!lowConst)
      return {};
    auto sliceTy = cast<hw::ArrayType>(
        cast<RefType>(op.getResult().getType()).getNestedType());
    auto wholeTy = cast<hw::ArrayType>(whole.getType());
    uint64_t low = lowConst.getValue().getZExtValue();
    uint64_t len = sliceTy.getNumElements();
    // Materialize the current slice value, recurse, then write the updated
    // elements back one by one.
    Value child = hw::ArraySliceOp::create(builder, loc, sliceTy, whole,
                                           op.getLowIndex());
    Value updated = rebuildInject(builder, loc, child, rest, value);
    if (!updated)
      return {};
    Value result = whole;
    unsigned wholeIdxWidth =
        std::max(1u, llvm::Log2_64_Ceil(wholeTy.getNumElements()));
    unsigned sliceIdxWidth =
        std::max(1u, llvm::Log2_64_Ceil(sliceTy.getNumElements()));
    for (uint64_t i = 0; i < len; ++i) {
      Value sliceIdx =
          hw::ConstantOp::create(builder, loc, APInt(sliceIdxWidth, i));
      Value elem = hw::ArrayGetOp::create(builder, loc, updated, sliceIdx);
      Value wholeIdx =
          hw::ConstantOp::create(builder, loc, APInt(wholeIdxWidth, low + i));
      result = hw::ArrayInjectOp::create(builder, loc, result, wholeIdx, elem);
    }
    return result;
  }
  if (auto op = dyn_cast<SigExtractOp>(step)) {
    auto lowConst = op.getLowBit().getDefiningOp<hw::ConstantOp>();
    if (!lowConst)
      return {};
    auto wholeWidth = hw::getBitWidth(whole.getType());
    auto sliceWidth = hw::getBitWidth(
        cast<RefType>(op.getResult().getType()).getNestedType());
    if (wholeWidth <= 0 || sliceWidth <= 0 || !isa<IntegerType>(whole.getType()))
      return {};
    uint64_t low = lowConst.getValue().getZExtValue();
    Value child = comb::ExtractOp::create(
        builder, loc, builder.getIntegerType(sliceWidth), whole, low);
    Value updated = rebuildInject(builder, loc, child, rest, value);
    if (!updated)
      return {};
    // comb.concat's FIRST operand is the MSBs (measured convention, see
    // Sig2RegPass).
    SmallVector<Value> parts;
    if (low + (uint64_t)sliceWidth < (uint64_t)wholeWidth)
      parts.push_back(comb::ExtractOp::create(
          builder, loc, builder.getIntegerType(wholeWidth - low - sliceWidth),
          whole, low + sliceWidth));
    parts.push_back(updated);
    if (low > 0)
      parts.push_back(comb::ExtractOp::create(
          builder, loc, builder.getIntegerType(low), whole, 0));
    if (parts.size() == 1)
      return parts[0];
    return comb::ConcatOp::create(builder, loc, parts);
  }
  return {};
}

/// Merge all drives in `block` that target the same root signal (directly or
/// through constant projections) into a single whole-signal drive, resolving
/// overlaps by program order.
static void mergeOrderedDrives(Block *block) {
  // Group the drives per root, preserving program order.
  llvm::MapVector<Value, SmallVector<DriveOp>> byRoot;
  for (auto driveOp : block->getOps<DriveOp>()) {
    SmallVector<Operation *> path;
    Value root = traceToRoot(driveOp.getSignal(), path);
    byRoot[root].push_back(driveOp);
  }

  for (auto &[root, drives] : byRoot) {
    if (drives.size() < 2)
      continue;
    // All drives must share the same delay; each projection chain must have
    // constant indices (checked during rebuilding; abandon on failure).
    Value delay = drives.front().getTime();
    if (llvm::any_of(drives, [&](DriveOp op) { return op.getTime() != delay; }))
      continue;

    auto lastDrive = drives.back();
    OpBuilder builder(lastDrive);
    auto loc = lastDrive.getLoc();

    // Resolve in program order. Seed with the signal's held value; if the
    // first drive is whole-signal and unconditional (the usual default
    // assignment), it replaces the seed outright and no probe is needed.
    Value cur;
    bool seeded = false;
    {
      SmallVector<Operation *> firstPath;
      if (traceToRoot(drives.front().getSignal(), firstPath) == root &&
          firstPath.empty() && !drives.front().getEnable()) {
        cur = drives.front().getValue();
        seeded = true;
      } else {
        cur = ProbeOp::create(builder, loc, root);
      }
    }

    bool abandoned = false;
    SmallVector<Operation *> newOps;
    for (auto [idx, driveOp] : llvm::enumerate(drives)) {
      if (seeded && idx == 0)
        continue;
      SmallVector<Operation *> path;
      traceToRoot(driveOp.getSignal(), path);
      Value updated =
          rebuildInject(builder, loc, cur, path, driveOp.getValue());
      if (!updated) {
        abandoned = true;
        break;
      }
      if (auto enable = driveOp.getEnable())
        cur = comb::MuxOp::create(builder, loc, enable, updated, cur, true);
      else
        cur = updated;
    }
    if (abandoned)
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "- Merged " << drives.size() << " ordered drives\n");
    DriveOp::create(builder, loc, root, cur, delay, Value{});
    for (auto driveOp : drives)
      driveOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Control Flow Removal
//===----------------------------------------------------------------------===//

namespace {
/// The main helper struct implementing control flow removal for a region.
struct CFRemover {
  CFRemover(Region &region) : region(region) {}
  void run();

  /// The region within which we are removing control flow.
  Region &region;
  /// The blocks in the region, sorted such that a block's predecessors appear
  /// in the list before the block itself.
  SmallVector<Block *> sortedBlocks;
  /// The dominance information for the region.
  DominanceInfo domInfo;
};
} // namespace

void CFRemover::run() {
  LLVM_DEBUG(llvm::dbgs() << "Removing control flow in " << region.getLoc()
                          << "\n");

  // Establish a topological order of the blocks in the region. Give up if we
  // detect a control flow cycle. Also take note of all YieldOps, such that we
  // can combine them into a single yield block later.
  SmallVector<YieldOp, 2> yieldOps;
  SmallPtrSet<Block *, 8> visitedBlocks, ipoSet;
  for (auto &block : region) {
    for (auto *ipoBlock : llvm::inverse_post_order_ext(&block, ipoSet)) {
      if (!llvm::all_of(ipoBlock->getPredecessors(), [&](auto *pred) {
            return visitedBlocks.contains(pred);
          })) {
        LLVM_DEBUG(llvm::dbgs() << "- Loop detected, giving up\n");
        return;
      }
      visitedBlocks.insert(ipoBlock);
      sortedBlocks.push_back(ipoBlock);
    }

    // Give up if there are any side-effecting ops in the region, except for
    // signal drives and probes, which this pass now handles explicitly. A
    // drive in a conditionally-executed block is rewritten into an
    // unconditionally-executed drive whose ENABLE carries the block's path
    // condition -- `llhd.drv ... if %cond` exists for exactly this. A probe
    // only reads, and within one region execution reordering it relative to
    // drives changes nothing: drives take effect a delta later, so probes
    // read pre-region signal state either way.
    //
    // This is what lets an `always_comb` that assigns a default and then
    // conditionally overrides fields of it -- CVA6's `ras` return-address
    // stack, the hpdcache control blocks -- flatten to a single block, which
    // in turn is the shape Sig2Reg's single-block transparency carve-out can
    // promote.
    for (auto &op : block) {
      if (isa<DriveOp, ProbeOp>(op))
        continue;
      if (!isMemoryEffectFree(&op)) {
        LLVM_DEBUG(llvm::dbgs() << "- Has side effects, giving up\n");
        return;
      }
    }

    // Check that we know what to do with all terminators.
    if (!isa<YieldOp, cf::BranchOp, cf::CondBranchOp>(block.getTerminator())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Has unsupported terminator "
                 << block.getTerminator()->getName() << ", giving up\n");
      return;
    }

    // Keep track of yield ops.
    if (auto yieldOp = dyn_cast<YieldOp>(block.getTerminator()))
      yieldOps.push_back(yieldOp);
  }

  // If there are multiple yield ops, factor them out into a single yield block.
  auto yieldOp = yieldOps[0];
  if (yieldOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "- Creating single yield block\n");
    OpBuilder builder(region.getContext());
    SmallVector<Location> locs(yieldOps[0].getNumOperands(), region.getLoc());
    auto *yieldBlock = builder.createBlock(&region, region.end(),
                                           yieldOps[0].getOperandTypes(), locs);
    sortedBlocks.push_back(yieldBlock);
    yieldOp =
        YieldOp::create(builder, region.getLoc(), yieldBlock->getArguments());
    for (auto yieldOp : yieldOps) {
      builder.setInsertionPoint(yieldOp);
      cf::BranchOp::create(builder, yieldOp.getLoc(), yieldBlock,
                           yieldOp.getOperands());
      yieldOp.erase();
    }
  }

  // Compute the dominance info for this region.
  domInfo = DominanceInfo(region.getParentOp());

  // Move operations into the entry block, replacing block arguments with
  // multiplexers as we go. The block order guarantees that we visit a block's
  // predecessors before we visit the block itself, so by the time a block is
  // merged every value its incoming branches carry, and every branch
  // condition between its dominator and itself, is already defined in the
  // entry block; only the (still intact) terminators are consulted.
  auto *entryBlock = sortedBlocks.front();
  for (auto *block : sortedBlocks) {
    if (!domInfo.isReachableFromEntry(block))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "- Merging block ";
      block->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Find the nearest common dominator block of all predecessors. Any block
    // arguments reaching the current block will only depend on control flow
    // decisions between this dominator block and the current block.
    auto *domBlock = block;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        domBlock = domInfo.findNearestCommonDominator(domBlock, pred);
    LLVM_DEBUG({
      llvm::dbgs() << "  - Common dominator: ";
      domBlock->printAsOperand(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    // Convert the block arguments into multiplexers nested by the branch
    // structure between the dominator and this block (see StructuredJoin).
    OpBuilder builder(entryBlock->getTerminator());
    if (block != entryBlock && block->getNumArguments() > 0) {
      auto mergedArgs = lowerJoinArguments(builder, domInfo, domBlock, block);
      assert(mergedArgs.size() == block->getNumArguments());
      for (auto [blockArg, mergedArg] :
           llvm::zip(block->getArguments(), mergedArgs))
        blockArg.replaceAllUsesWith(mergedArg);
    }

    // Fold the path condition from the entry block into the enable of every
    // drive in this block: the drive used to execute only when control
    // reached the block, and after the merge it executes always. The
    // condition values materialize at the entry terminator, which precedes
    // the spliced ops; the AND itself is built at the drive, after its
    // original enable's def. The condition is built by the same structured
    // walk as the block arguments, so a branch both of whose arms reach the
    // block does not appear in the enable.
    if (block != entryBlock) {
      SmallVector<DriveOp> drives(block->getOps<DriveOp>());
      if (!drives.empty()) {
        auto decision = getReachCondition(builder, domInfo, entryBlock, block);
        if (!decision.isTrue()) {
          for (auto driveOp : drives) {
            OpBuilder driveBuilder(driveOp);
            Value cond =
                decision.materialize(driveBuilder, driveOp.getLoc());
            if (auto oldEnable = driveOp.getEnable())
              cond = driveBuilder.createOrFold<comb::AndOp>(
                  driveOp.getLoc(), oldEnable, cond, /*twoState=*/true);
            driveOp.getEnableMutable().assign(cond);
          }
        }
      }
    }

    // Move all ops except for the terminator into the entry block.
    if (block != entryBlock)
      entryBlock->getOperations().splice(--entryBlock->end(),
                                         block->getOperations(), block->begin(),
                                         --block->end());
  }

  // Move the yield op into the entry block, replacing the original terminator.
  if (yieldOp != entryBlock->getTerminator()) {
    yieldOp->moveBefore(entryBlock->getTerminator());
    entryBlock->getTerminator()->erase();
  }

  // Remove all blocks except for the entry block. We first clear all operations
  // in the blocks such that the blocks have no more uses in branch ops. Then we
  // remove the blocks themselves in a second pass.
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->clear();
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->erase();

  return;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct RemoveControlFlowPass
    : public llhd::impl::RemoveControlFlowPassBase<RemoveControlFlowPass> {
  void runOnOperation() override;
};
} // namespace

void RemoveControlFlowPass::runOnOperation() {
  for (auto op : getOperation().getOps<CombinationalOp>()) {
    CFRemover(op.getBody()).run();
    // If the body is (now) a single block, its drives execute in a defined
    // program order; fold each signal's ordered drives into one.
    if (op.getBody().hasOneBlock())
      mergeOrderedDrives(&op.getBody().front());
  }
}
