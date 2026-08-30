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

private:
  llvm::PointerIntPair<Value, 2> pair;
};
} // namespace

/// Compute the branch decisions that cause control to flow from the dominator
/// to the target block.
///
/// TODO: This eagerly aggregates all control flow decisions. It may be more
/// efficient to first determine which blocks lie in between dominator and
/// target, and then only check that we are not taking decisions that cause
/// control flow to *leave* that set of blocks.
static Condition getBranchDecisionsFromDominatorToTarget(
    OpBuilder &builder, Block *dominator, Block *target,
    SmallDenseMap<std::pair<Block *, Block *>, Condition> &decisions) {
  if (auto decision = decisions.lookup({dominator, target}))
    return decision;

  SmallPtrSet<Block *, 8> visitedBlocks;
  visitedBlocks.insert(dominator); // stop at the dominator
  if (auto &decision = decisions[{dominator, dominator}]; !decision)
    decision = Condition(true);

  // Traverse the blocks in inverse post order. This ensures that we are
  // visiting all of a block's predecessors before we visit the block itself.
  // This allows us to first compute the decision leading control flow to each
  // of the predecessors, such that the current block can then just combine the
  // predecessor decisions.
  for (auto *block : llvm::inverse_post_order_ext(target, visitedBlocks)) {
    auto merged = Condition(false);
    for (auto *pred : block->getPredecessors()) {
      auto predDecision = decisions.lookup({dominator, pred});
      assert(predDecision);
      if (pred->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(pred->getTerminator());
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          merged = merged.orWith(predDecision, builder);
        } else {
          auto cond = Condition(condBr.getCondition());
          if (condBr.getFalseDest() == block)
            cond = cond.inverted(builder);
          merged = merged.orWith(cond.andWith(predDecision, builder), builder);
        }
      } else {
        merged = merged.orWith(predDecision, builder);
      }
    }
    assert(merged);
    decisions.insert({{dominator, block}, merged});
  }

  return decisions.lookup({dominator, target});
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
  // predecessors before we visit the block itself.
  SmallDenseMap<std::pair<Block *, Block *>, Condition> decisionCache;
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

    // Convert the block arguments into multiplexers.
    SmallVector<Block *, 4> reachablePreds;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        reachablePreds.push_back(pred);

    llvm::stable_sort(reachablePreds, [&](Block *a, Block *b) {
      return a == domBlock && b != domBlock;
    });

    OpBuilder builder(entryBlock->getTerminator());
    SmallVector<Value> mergedArgs;
    SmallPtrSet<Block *, 4> seenPreds;
    for (auto *pred : reachablePreds) {
      // A block may be listed multiple times in the predecessors.
      if (!seenPreds.insert(pred).second)
        continue;

      // Only consider values coming from reachable predecessors.
      if (!domInfo.isReachableFromEntry(pred))
        continue;

      // Helper function to create a multiplexer between the current
      // `mergedArgs` and a new set of `args`, where the new args are picked if
      // `cond` is true and control flows from `domBlock` to `pred`. The
      // condition may be null, in which case the mux will directly use the
      // branch decisions that lead from `domBlock` to `pred`.
      auto mergeArgs = [&](ValueRange args, Condition cond, bool invCond) {
        if (mergedArgs.empty()) {
          mergedArgs = args;
          return;
        }
        auto decision = getBranchDecisionsFromDominatorToTarget(
            builder, domBlock, pred, decisionCache);
        if (cond) {
          if (invCond)
            cond = cond.inverted(builder);
          decision = decision.andWith(cond, builder);
        }
        for (auto [mergedArg, arg] : llvm::zip(mergedArgs, args)) {
          if (decision.isTrue())
            mergedArg = arg;
          else if (decision.isFalse())
            continue;
          else
            mergedArg = builder.createOrFold<comb::MuxOp>(
                arg.getLoc(), decision.materialize(builder, arg.getLoc()), arg,
                mergedArg);
        }
      };

      // Handle the different terminators that we support.
      if (auto condBrOp = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (condBrOp.getTrueDest() == condBrOp.getFalseDest()) {
          // Both destinations lead to the current block. Insert a mux to
          // collapse the destination operands and then treat this as an
          // unconditional branch to the current block.
          LLVM_DEBUG(llvm::dbgs() << "  - Both from " << condBrOp << "\n");
          SmallVector<Value> mergedOperands;
          mergedOperands.reserve(block->getNumArguments());
          for (auto [trueArg, falseArg] :
               llvm::zip(condBrOp.getTrueDestOperands(),
                         condBrOp.getFalseDestOperands())) {
            mergedOperands.push_back(builder.createOrFold<comb::MuxOp>(
                trueArg.getLoc(), condBrOp.getCondition(), trueArg, falseArg));
          }
          mergeArgs(mergedOperands, Value{}, false);
        } else if (condBrOp.getTrueDest() == block) {
          // The branch leads to the current block if the condition is true.
          LLVM_DEBUG(llvm::dbgs() << "  - True from " << condBrOp << "\n");
          mergeArgs(condBrOp.getTrueDestOperands(), condBrOp.getCondition(),
                    false);
        } else {
          // The branch leads to the current block if the condition is false.
          LLVM_DEBUG(llvm::dbgs() << "  - False from " << condBrOp << "\n");
          mergeArgs(condBrOp.getFalseDestOperands(), condBrOp.getCondition(),
                    true);
        }
      } else {
        auto brOp = cast<cf::BranchOp>(pred->getTerminator());
        LLVM_DEBUG(llvm::dbgs() << "  - From " << brOp << "\n");
        mergeArgs(brOp.getDestOperands(), Value{}, false);
      }
    }
    for (auto [blockArg, mergedArg] :
         llvm::zip(block->getArguments(), mergedArgs))
      blockArg.replaceAllUsesWith(mergedArg);

    // Fold the path condition from the entry block into the enable of every
    // drive in this block: the drive used to execute only when control
    // reached the block, and after the merge it executes always. The
    // condition values materialize at the entry terminator, which precedes
    // the spliced ops; the AND itself is built at the drive, after its
    // original enable's def.
    if (block != entryBlock) {
      SmallVector<DriveOp> drives(block->getOps<DriveOp>());
      if (!drives.empty()) {
        auto decision = getBranchDecisionsFromDominatorToTarget(
            builder, entryBlock, block, decisionCache);
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
