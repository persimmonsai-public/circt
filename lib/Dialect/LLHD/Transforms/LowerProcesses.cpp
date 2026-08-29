//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-lower-processes"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_LOWERPROCESSESPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;

namespace {
struct Lowering {
  Lowering(ProcessOp processOp, bool assumeCompleteSensitivity)
      : processOp(processOp),
        assumeCompleteSensitivity(assumeCompleteSensitivity) {}
  void lower();
  bool matchControlFlow();
  void markObservedValues();
  bool allOperandsObserved();
  bool isObserved(Value value);

  bool canReRootAtWaitDest();
  bool checkNoValuesLiveAcrossWait();

  ProcessOp processOp;
  /// See the pass option of the same name.
  bool assumeCompleteSensitivity;
  WaitOp waitOp;
  /// Set when the process is combinational but does not have the shape where
  /// entry and wait converge. See `canReRootAtWaitDest`.
  bool reRootAtWaitDest = false;
  SmallDenseSet<Value> observedValues;
};
} // namespace

void Lowering::lower() {
  // Ensure that the process describes combinational logic.
  if (!matchControlFlow())
    return;
  markObservedValues();
  // An incomplete sensitivity list means the process does not re-evaluate on
  // every value it reads. Synthesis reads that as combinational logic anyway;
  // simulation does not. Only skip the check when the caller has opted in.
  if (!assumeCompleteSensitivity && !allOperandsObserved())
    return;
  LLVM_DEBUG(llvm::dbgs() << "Lowering process " << processOp.getLoc() << "\n");

  // Replace the process.
  OpBuilder builder(processOp);
  auto executeOp = CombinationalOp::create(builder, processOp.getLoc(),
                                           processOp.getResultTypes());
  executeOp.getRegion().takeBody(processOp.getBody());
  processOp.replaceAllUsesWith(executeOp);
  processOp.erase();
  processOp = {};

  // If the process carries its outputs around the wait loop as block
  // arguments, the entry block only supplies their pre-first-event values and
  // the real computation hangs off the wait's destination. Make that
  // destination the region's entry block, so the initialisation path becomes
  // unreachable and `simplifyRegions` below drops it. Without this the
  // combinational op would yield the initial constants and the whole body
  // would be deleted as unreachable.
  if (reRootAtWaitDest) {
    auto &region = executeOp.getRegion();
    auto &blocks = region.getBlocks();
    if (waitOp.getDestOperands().empty()) {
      blocks.splice(blocks.begin(), blocks, waitOp.getDest()->getIterator());
    } else {
      // The destination needs arguments, so it cannot itself be the entry.
      // Prepend a block that jumps to it with the wait's own destination
      // operands, all of which are defined outside the process.
      auto *newEntry = new Block();
      blocks.push_front(newEntry);
      OpBuilder entryBuilder(newEntry, newEntry->begin());
      cf::BranchOp::create(entryBuilder, waitOp.getLoc(), waitOp.getDest(),
                           waitOp.getDestOperands());
    }
  }

  // Replace the `llhd.wait` with an `llhd.yield`.
  builder.setInsertionPoint(waitOp);
  YieldOp::create(builder, waitOp.getLoc(), waitOp.getYieldOperands());
  waitOp.erase();

  // Simplify the execute op body region since disconnecting the control flow
  // loop through the wait op has potentially created unreachable blocks.
  IRRewriter rewriter(builder);
  (void)simplifyRegions(rewriter, executeOp->getRegions());
}

/// Check that the process' entry block trivially joins a control flow loop
/// immediately after the wait op.
bool Lowering::matchControlFlow() {
  // Ensure that there is only a single wait op in the process and that it has
  // no destination operands.
  for (auto &block : processOp.getBody()) {
    // Processes that can terminate are not combinational. Lowering them to
    // `llhd.combinational` would leave an illegal `llhd.halt` terminator in the
    // new region.
    if (isa<HaltOp>(block.getTerminator())) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                              << ": contains halt terminator\n");
      return false;
    }
    if (auto op = dyn_cast<WaitOp>(block.getTerminator())) {
      if (waitOp) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                                << ": multiple wait ops\n");
        return false;
      }
      waitOp = op;
    }
  }
  if (!waitOp) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                            << ": no wait op\n");
    return false;
  }
  if (!waitOp.getDestOperands().empty()) {
    // The destination operands can still be supplied by a synthetic entry
    // block, provided they are defined outside the process (see
    // canReRootAtWaitDest).
    if (!canReRootAtWaitDest()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                              << ": wait op has destination operands\n");
      return false;
    }
    reRootAtWaitDest = true;
  }
  if (waitOp.getDelay()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                            << ": wait op has delay\n");
    return false;
  }

  // Helper function to skip across empty blocks with only a single successor.
  auto skipToMergePoint = [&](Block *block) -> std::pair<Block *, ValueRange> {
    ValueRange operands;
    while (auto branchOp = dyn_cast<cf::BranchOp>(block->getTerminator())) {
      if (llvm::any_of(block->without_terminator(),
                       [](auto &op) { return !isMemoryEffectFree(&op); }))
        break;
      block = branchOp.getDest();
      operands = branchOp.getDestOperands();
      if (std::distance(block->pred_begin(), block->pred_end()) > 1)
        break;
      if (!operands.empty())
        break;
    }
    return {block, operands};
  };

  // Ensure that the entry block and wait op converge on the same block.
  // Re-rooting makes this moot: the entry path is dropped outright.
  if (reRootAtWaitDest)
    return checkNoValuesLiveAcrossWait();

  auto &entry = processOp.getBody().front();
  auto [entryMergeBlock, entryMergeArgs] = skipToMergePoint(&entry);
  auto [waitMergeBlock, waitMergeArgs] = skipToMergePoint(waitOp.getDest());
  if (entryMergeBlock != waitMergeBlock) {
    if (!canReRootAtWaitDest()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping process " << processOp.getLoc()
                 << ": control from entry and wait does not converge\n");
      return false;
    }
    reRootAtWaitDest = true;
  }

  // Helper function to check if two values are equivalent.
  auto areValuesEquivalent = [](std::tuple<Value, Value> values) {
    auto [a, b] = values;
    if (a == b)
      return true;
    auto *opA = a.getDefiningOp();
    auto *opB = b.getDefiningOp();
    if (!opA || !opB)
      return false;
    return OperationEquivalence::isEquivalentTo(
        opA, opB, OperationEquivalence::IgnoreLocations);
  };

  // Ensure that the entry block and wait op converge with equivalent block
  // arguments.
  if (!reRootAtWaitDest &&
      !llvm::all_of(llvm::zip(entryMergeArgs, waitMergeArgs),
                    areValuesEquivalent)) {
    if (!canReRootAtWaitDest()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping process " << processOp.getLoc()
                              << ": control from entry and wait converges with "
                                 "different block arguments\n");
      return false;
    }
    reRootAtWaitDest = true;
  }

  return checkNoValuesLiveAcrossWait();
}

/// Ensure that no values are live across the wait op.
bool Lowering::checkNoValuesLiveAcrossWait() {
  Liveness liveness(processOp);
  for (auto value : liveness.getLiveOut(waitOp->getBlock())) {
    if (value.getParentRegion()->isProperAncestor(&processOp.getBody()))
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Skipping process " << processOp.getLoc() << ": value ";
      value.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << " live across wait\n";
    });
    return false;
  }
  return true;
}

/// Check whether a process that does NOT have the "entry and wait converge"
/// shape is nevertheless combinational, by carrying its outputs around the
/// wait loop as block arguments.
///
/// This is the shape older Verilog produces for `always @(*)` with case
/// statements assigning to regs -- pervasive in OpenC910, where it accounts
/// for 912 of 1,441 processes:
///
///     ^bb0:                                   // entry, pre-first-event values
///       cf.br ^bb1(%c0, %false : i66, i1)
///     ^bb1(%v0: i66, %v1: i1):                // loop header
///       llhd.wait yield (%v0, %v1), (%a, %b), ^bb2
///     ^bb2:                                   // the real body
///       ...
///       cf.br ^bb1(%computed0, %computed1)
///
/// The header's block arguments are the process' outputs. The entry block
/// only supplies their value before the first input event; every path through
/// the body branches back with freshly computed values. So the process IS a
/// function of its inputs, and the combinational form is obtained by making
/// the wait's destination the entry block -- dropping the initialisation path.
///
/// That is only sound when the loop-carried arguments feed nothing but the
/// wait itself. If the body READ them, a path that branched back without
/// reassigning would preserve the previous value -- a latch, i.e. real state,
/// which is not combinational and must not be lowered this way. Requiring the
/// arguments to be used solely as wait yield operands rules that out.
///
/// Dropping the initialisation path changes the value yielded before the
/// first input event, from the entry constants to `f(inputs)`. For synthesis
/// that is the correct reading -- combinational logic has no "initial" value
/// -- and it also matches Verilog, where an `always @(*)` block does execute
/// at time zero.
bool Lowering::canReRootAtWaitDest() {
  // The loop-carried values must not be read by the body.
  for (auto arg : waitOp->getBlock()->getArguments())
    for (auto &use : arg.getUses())
      if (use.getOwner() != waitOp.getOperation())
        return false;

  // If the destination takes arguments, a synthetic entry block has to supply
  // them, so every operand must already dominate the process -- i.e. be
  // defined outside it. An operand computed inside the body cannot be used to
  // start the body.
  for (auto operand : waitOp.getDestOperands())
    if (!operand.getParentRegion()->isProperAncestor(&processOp.getBody()))
      return false;

  // The entry block must carry nothing but its branch: once it is unreachable
  // it gets deleted, and anything it defined would leave dangling uses.
  auto &entry = processOp.getBody().front();
  if (!entry.without_terminator().empty())
    return false;
  if (!isa<cf::BranchOp>(entry.getTerminator()))
    return false;

  // Re-rooting at the entry block itself would be a no-op loop.
  if (waitOp.getDest() == &entry)
    return false;

  return true;
}

/// Mark values the process observes that are defined outside the process.
void Lowering::markObservedValues() {
  SmallVector<Value> worklist;
  auto markObserved = [&](Value value) {
    if (observedValues.insert(value).second)
      worklist.push_back(value);
  };

  for (auto value : waitOp.getObserved())
    if (value.getParentRegion()->isProperAncestor(&processOp.getBody()))
      markObserved(value);

  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    auto *op = value.getDefiningOp();
    if (!op)
      continue;

    // Look through probe ops to mark the probe signal as well, just in case
    // there may be multiple probes of the same signal.
    if (auto probeOp = dyn_cast<ProbeOp>(op))
      markObserved(probeOp.getSignal());

    // Look through operations that simply reshape incoming values into an
    // aggregate form from which any changes remain apparent.
    if (isa<hw::ArrayCreateOp, hw::StructCreateOp, comb::ConcatOp,
            hw::BitcastOp>(op))
      for (auto operand : op->getOperands())
        markObserved(operand);
  }
}

/// Ensure that any value defined outside the process that is used inside the
/// process is derived entirely from an observed value.
bool Lowering::allOperandsObserved() {
  // Collect all ancestor regions such that we can easily check if a value is
  // defined outside the process.
  SmallPtrSet<Region *, 4> properAncestors;
  for (auto *region = processOp->getParentRegion(); region;
       region = region->getParentRegion())
    properAncestors.insert(region);

  // Walk all operations under the process and check each operand.
  auto walkResult = processOp.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      // We only care about values defined outside the process.
      if (!properAncestors.count(operand.getParentRegion()))
        continue;

      // If the value is observed, all is well.
      if (isObserved(operand))
        continue;

      // Otherwise complain and abort.
      LLVM_DEBUG({
        llvm::dbgs() << "Skipping process " << processOp.getLoc()
                     << ": unobserved value ";
        operand.print(llvm::dbgs(), OpPrintingFlags().skipRegions());
        llvm::dbgs() << "\n";
      });
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !walkResult.wasInterrupted();
}

/// Check if a value is observed by the wait op, or all its operands are only
/// derived from observed values.
bool Lowering::isObserved(Value value) {
  // Check if the value is trivially observed.
  if (observedValues.contains(value))
    return true;

  // Otherwise get the operation that defines it such that we can check if the
  // value is derived from purely observed values. If it isn't define by an op,
  // the value is unobserved.
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  // Otherwise visit all ops in the fan-in cone and ensure that they are
  // observed. If any value is unobserved, immediately return false.
  SmallDenseSet<Operation *> seenOps;
  SmallVector<Operation *> worklist;
  seenOps.insert(defOp);
  worklist.push_back(defOp);
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();

    // Give up on ops with nested regions.
    if (op->getNumRegions() != 0)
      return false;

    // Otherwise check that all operands are observed. If we haven't seen an
    // operand before, and it is not a signal, add it to the worklist to be
    // checked.
    for (auto operand : op->getOperands()) {
      if (observedValues.contains(operand))
        continue;
      if (isa<RefType>(operand.getType()))
        return false;
      auto *defOp = operand.getDefiningOp();
      if (!defOp || !isMemoryEffectFree(defOp))
        return false;
      if (seenOps.insert(defOp).second)
        worklist.push_back(defOp);
    }
  }

  // If we arrive at this point, we weren't able to reach an unobserved value.
  // Therefore we consider this value derived from only observed values.
  observedValues.insert(value);
  return true;
}

namespace {
struct LowerProcessesPass
    : public llhd::impl::LowerProcessesPassBase<LowerProcessesPass> {
  using LowerProcessesPassBase::LowerProcessesPassBase;
  void runOnOperation() override;
};
} // namespace

void LowerProcessesPass::runOnOperation() {
  SmallVector<ProcessOp> processOps(getOperation().getOps<ProcessOp>());
  for (auto processOp : processOps)
    Lowering(processOp, assumeCompleteSensitivity).lower();
}
