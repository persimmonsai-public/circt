//===- Sig2RegPass.cpp - Implement the Sig2Reg Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to promote LLHD signals to SSA values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "llhd-sig2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SIG2REG
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Represents an offset of an interval relative to a root interval. All values
/// describe number of bits, not elements.
struct Offset {
  Offset(uint64_t min, uint64_t max, ArrayRef<Value> dynamic)
      : min(min), max(max), dynamic(dynamic) {}

  Offset(uint64_t idx) : min(idx), max(idx) {}

  // The lower bound of the offset known statically.
  uint64_t min = 0;
  // The upper bound of the offset known statically.
  uint64_t max = -1;
  // A list of SSA values used to compute the final offset.
  SmallVector<Value> dynamic;

  /// Returns if we know the exact offset statically.
  bool isStatic() const { return min == max; }
};

/// Represents an alias interval within a root interval that is written to or
/// read from. All values refer to number of bits, not elements.
struct Interval {
  Interval(const Offset &low, uint64_t bitwidth, Value value,
           Block *block = nullptr, llhd::TimeAttr delay = llhd::TimeAttr())
      : low(low), bitwidth(bitwidth), value(value), block(block),
        delay(delay) {}

  // The offset of the interval relative to the root interval (i.e. all the bits
  // of the original signal).
  Offset low;
  // The width of the interval.
  uint64_t bitwidth;
  // The value written to this interval or the OpResult of a read.
  Value value;
  // The block the originating probe/drive lived in. Equal to the signal's
  // own block except for reads/writes reached via the single-block
  // `llhd.combinational` transparency carve-out above, in which case this is
  // that op's body block instead.
  Block *block;
  // The delay with which the value is written.
  llhd::TimeAttr delay;
};

/// A narrowed slice of a whole-signal drive: the range [bitOffset,
/// bitOffset+bitwidth) of the overall signal that this slice genuinely
/// writes. The caller is responsible for building the actual replacement
/// value (by extracting this same range back out of the drive's own
/// value -- see the two call sites below), this struct only records WHERE.
struct NarrowedWrite {
  uint64_t bitOffset;
  uint64_t bitwidth;
};

/// Unwraps a clock-enable self-feedback mux, if present: a clock-enable
/// register is `Deseq`'d into `next = comb.mux(enable, <new-value
/// structure>, <self-feedback>)` ("hold current value when disabled")
/// rather than the new-value structure directly. The enable condition
/// doesn't change WHICH position a register is responsible for, only
/// whether it updates that position or holds it, and either way extracting
/// the position back out of the register's OWN result (built by the
/// caller) reflects the correct current content regardless of which
/// branch fired -- so this just recurses into whichever mux operand is
/// NOT the self-feedback value, when `selfFeedback` is non-null and such a
/// mux is present; otherwise returns `feeder` unchanged.
static Value unwrapClockEnable(Value feeder, Value selfFeedback) {
  if (!selfFeedback)
    return feeder;
  if (auto muxOp = feeder.getDefiningOp<comb::MuxOp>()) {
    if (muxOp.getTrueValue() == selfFeedback)
      return muxOp.getFalseValue();
    if (muxOp.getFalseValue() == selfFeedback)
      return muxOp.getTrueValue();
  }
  return feeder;
}

/// Attempt to decompose a whole-signal drive into a set of narrower,
/// per-element writes by inspecting a *value feeding it* (typically a
/// `seq.firreg`'s `next` operand, i.e. `feeder` is NOT the driven value
/// itself -- see the DriveOp handling below, and `unwrapClockEnable` above)
/// for a chain of `hw.array_inject`s over a shared base that is itself a
/// probe of `sigOp`. This is the shape `Deseq` produces for a
/// `for (genvar i...)` per-element clocked write: each resulting
/// `seq.firreg`'s "next" value is nominally the FULL array, but is
/// actually the array probed at the START of this evaluation
/// (`llhd.prb %sigOp`) with only its own element(s) replaced -- multiple
/// such firregs, each individually reported as a full-width write, look
/// "potentially overlapping" to isPromotable()'s conservative check even
/// though they write disjoint elements, because the untouched portion of
/// each is a faithful, unconditional copy of the SAME prior signal value
/// every sibling write (and the overall reconstruction's `preVal`) already
/// accounts for -- narrowing here just makes that already-true fact
/// visible to the interval algorithm instead of reporting a conservative
/// upper bound.
///
/// Returns the peeled writes on success (may be more than one, for a chain
/// injecting several positions), or std::nullopt if `feeder` doesn't have
/// this shape -- in which case the caller should fall back to a single
/// full-width interval, exactly as before this narrowing existed.
static std::optional<SmallVector<NarrowedWrite>>
tryNarrowArrayInjectChain(Value feeder, llhd::SignalOp sigOp) {
  SmallVector<NarrowedWrite> narrowed;
  Value cur = feeder;
  while (auto injectOp = cur.getDefiningOp<hw::ArrayInjectOp>()) {
    auto constIdx = injectOp.getIndex().getDefiningOp<hw::ConstantOp>();
    if (!constIdx)
      return std::nullopt;

    auto arrayTy = cast<hw::ArrayType>(injectOp.getInput().getType());
    auto elemBw = hw::getBitWidth(injectOp.getElement().getType());
    if (elemBw <= 0)
      return std::nullopt;

    // Element 0 occupies the LOW bits of the flattened bitcast
    // representation. Verified against upstream CIRCT's own ops (the
    // authoritative convention, since the resolved value assembled from
    // these offsets is consumed by upstream array_get/bitcast lowering):
    // `hw.array_get (hw.bitcast %x : (i64) -> !hw.array<2xi32>)[0]`
    // exports as `x[31:0]`. The earlier revision of this code claimed the
    // opposite ("element 0 in the HIGH bits") from reading
    // `hw.aggregate_constant [1, 2] : !hw.array<2xi32>` == 0x1_00000002 --
    // but that experiment was misread: aggregate_constant (like
    // array_create) lists elements HIGHEST-INDEX-FIRST, so the leading
    // `1` is element 1 (high bits) and element 0 is the trailing `2`
    // (low bits), consistent with element-0-at-LSB. The flipped offset
    // this produced made reads of array entry i resolve to entry N-1-i's
    // storage (a real, silent miscompilation of ibex_fetch_fifo's FIFO,
    // caught by a depth-2 bounded equivalence check against the RTL).
    uint64_t index = constIdx.getValue().getZExtValue();
    uint64_t offset = index * elemBw;
    narrowed.push_back({offset, static_cast<uint64_t>(elemBw)});
    cur = injectOp.getInput();
  }

  if (narrowed.empty())
    return std::nullopt;

  auto probeOp = cur.getDefiningOp<llhd::ProbeOp>();
  if (!probeOp || probeOp.getSignal() != sigOp.getResult())
    return std::nullopt;

  return narrowed;
}

/// The struct counterpart of `tryNarrowArrayInjectChain`, for the single most
/// common shape in struct-heavy RTL: an `always_comb` that assigns a default
/// to a packed struct and then overrides individual fields, which elaborates
/// into a chain of `hw.struct_inject`s over a shared `llhd.prb %sigOp`.
///
/// CVA6's `load_unit.req_port_o` is the motivating case. It is a 117-bit
/// struct with EIGHT drives: seven per-field ones (`data_we`, `data_wdata`,
/// `cbo_op`, ...) and one whole-struct drive whose feeder is a mux tree over
/// `struct_inject` chains touching only `data_req`, `kill_req`, `tag_valid`,
/// `data_be`, `data_size` and `valid`. Those two sets are disjoint, but the
/// whole-struct drive is reported at full width, so all eight collide and the
/// signal is refused. Narrowing it to the six injected fields makes the
/// already-true disjointness visible to the interval algorithm.
///
/// FIELD OFFSETS ARE THE OPPOSITE OF ARRAY OFFSETS. Measured against
/// `hw-aggregate-to-comb`, which is what actually flattens these types:
/// `!hw.struct<a:i2, b:i6, c:i4>` bitcast from an i12 puts `a` at bit 10, `b`
/// at 4 and `c` at 0 -- the FIRST field occupies the HIGH bits, while array
/// element 0 occupies the LOW bits. So a field's offset is the summed width
/// of the fields AFTER it. Do not "simplify" this to match the array case.
static std::optional<SmallVector<NarrowedWrite>>
tryNarrowStructInjectChain(Value feeder, llhd::SignalOp sigOp) {
  SmallVector<NarrowedWrite> narrowed;
  Value cur = feeder;
  while (auto injectOp = cur.getDefiningOp<hw::StructInjectOp>()) {
    auto structTy = hw::type_cast<hw::StructType>(injectOp.getInput().getType());
    auto index = structTy.getFieldIndex(injectOp.getFieldNameAttr());
    if (!index)
      return std::nullopt;

    auto elements = structTy.getElements();
    auto fieldBw = hw::getBitWidth(elements[*index].type);
    if (fieldBw <= 0)
      return std::nullopt;

    uint64_t offset = 0;
    for (size_t i = *index + 1, e = elements.size(); i < e; ++i) {
      auto bw = hw::getBitWidth(elements[i].type);
      if (bw < 0)
        return std::nullopt;
      offset += bw;
    }

    narrowed.push_back({offset, static_cast<uint64_t>(fieldBw)});
    cur = injectOp.getInput();
  }

  if (narrowed.empty())
    return std::nullopt;

  // Same base requirement as the array case: the chain has to bottom out at a
  // probe of THIS signal, which is what makes every untouched field a faithful
  // pass-through that `preVal` already accounts for. A chain over anything
  // else is writing a genuinely new value and must keep its full width.
  auto probeOp = cur.getDefiningOp<llhd::ProbeOp>();
  if (!probeOp || probeOp.getSignal() != sigOp.getResult())
    return std::nullopt;

  return narrowed;
}

/// The bit-vector counterpart of `tryNarrowArrayInjectChain`, for a
/// non-aggregate (plain integer) signal: `Deseq` represents a per-position
/// clocked bitfield write as `comb.concat(..., <new bits>, ...)` where the
/// OTHER operands are `comb.extract`s of the shared `llhd.prb %sigOp`
/// (faithfully passing through whatever those bit ranges already held).
/// Every operand must be classified (either a probe-extract, contributing
/// no interval since `preVal` already covers it, or a genuine write) for
/// this to succeed -- a concat with any operand that is neither falls back
/// to std::nullopt, same as the array case.
static std::optional<SmallVector<NarrowedWrite>>
tryNarrowBitConcatChain(Value feeder, llhd::SignalOp sigOp) {
  auto concatOp = feeder.getDefiningOp<comb::ConcatOp>();
  if (!concatOp)
    return std::nullopt;

  SmallVector<NarrowedWrite> narrowed;
  // comb.concat operands are MSB-first (verified empirically against this
  // exact construct: `comb.extract %prb from 1 : (i3) -> i2` (bits [2:1])
  // followed by the new bit becomes `comb.concat %that, %newBit`, i.e. the
  // higher-order extract comes first) -- walk operands accumulating from
  // the top bit down.
  uint64_t bitWidth = hw::getBitWidth(concatOp.getType());
  uint64_t highBit = bitWidth;
  for (Value operand : concatOp.getInputs()) {
    uint64_t width = hw::getBitWidth(operand.getType());
    uint64_t lowBit = highBit - width;
    highBit = lowBit;

    if (auto extractOp = operand.getDefiningOp<comb::ExtractOp>()) {
      if (auto probeOp = extractOp.getInput().getDefiningOp<llhd::ProbeOp>()) {
        if (probeOp.getSignal() == sigOp.getResult() &&
            extractOp.getLowBit() == lowBit) {
          continue; // Faithful pass-through -- no interval needed here.
        }
      }
    }

    narrowed.push_back({lowBit, width});
  }

  if (narrowed.empty())
    return std::nullopt;
  return narrowed;
}

/// Narrow a drive's feeder to the bit ranges it actually writes, looking
/// through mux trees.
///
/// `tryNarrowArrayInjectChain`/`tryNarrowBitConcatChain` each require the
/// pass-through shape to sit DIRECTLY on the feeder. That covers the case
/// where the hold path is a clock-enable self-feedback mux, which
/// `unwrapClockEnable` peels first. It does not cover RTL that expresses the
/// hold by re-injecting into the signal's own probe on every arm, which is
/// what OpenC910's per-entry gated-clock register banks do:
///
///     %58 = comb.mux %57, %53, %55                  // no self-feedback arm
///     %53 = hw.array_inject %74[%false], %52        // element 0
///     %55 = comb.mux %54, %51, %50                  // ... also element 0
///
/// Every leaf writes the same element, so the drive's real interval is that
/// element -- but with a mux on top neither helper matches and the drive is
/// reported at full width, which then makes two sibling drives of disjoint
/// elements look like they collide. That rejected 65 signals on OpenC910 and
/// blackboxed 55 modules.
///
/// Recursing into both arms and unioning is sound because the union is an
/// upper bound on what either arm writes, and the caller reconstructs each
/// range from the register's own RESULT -- whichever arm fired, the result
/// holds the value that range ended up with. Any arm that cannot be narrowed
/// forces the whole feeder back to full width, and overlapping-but-unequal
/// ranges are refused outright rather than merged, keeping this a pure
/// narrowing of the existing conservative answer.
static std::optional<SmallVector<NarrowedWrite>>
tryNarrowFeeder(Value feeder, llhd::SignalOp sigOp, Value selfFeedback,
                unsigned depth = 0) {
  feeder = unwrapClockEnable(feeder, selfFeedback);

  // A feeder that is exactly a probe of the signal being driven is a pure
  // hold: it writes nothing. Without this the arm poisons the whole mux
  // union below and the drive falls back to full width, which is what made
  // two drives of DISJOINT bits look identical. OpenC910's `ct_fifo` writes
  // one bit of a 2-bit valid vector per entry and expresses "no write this
  // cycle" exactly this way:
  //
  //     %38 = comb.mux %34, %35, %37
  //       %35 = comb.concat %32, %true        // bit 0 written
  //       %37 = comb.mux %36, %33, %70
  //          %70 = llhd.prb %fifo_entry_vld   // <- writes nothing
  //
  // Restricted to a probe of THIS signal; a probe of any other signal is a
  // real value being written and must keep its full-width interval.
  if (auto probeOp = feeder.getDefiningOp<llhd::ProbeOp>())
    if (probeOp.getSignal() == sigOp.getResult())
      return SmallVector<NarrowedWrite>{};

  if (auto narrowed = tryNarrowArrayInjectChain(feeder, sigOp))
    return narrowed;
  if (auto narrowed = tryNarrowStructInjectChain(feeder, sigOp))
    return narrowed;
  if (auto narrowed = tryNarrowBitConcatChain(feeder, sigOp))
    return narrowed;

  // Bound the recursion: these mux trees are shallow in practice, and a cap
  // keeps a pathological input from costing more than the promotion is worth.
  static constexpr unsigned kMaxMuxDepth = 8;
  if (depth >= kMaxMuxDepth)
    return std::nullopt;

  auto muxOp = feeder.getDefiningOp<comb::MuxOp>();
  if (!muxOp)
    return std::nullopt;

  auto lhs = tryNarrowFeeder(muxOp.getTrueValue(), sigOp, selfFeedback,
                             depth + 1);
  if (!lhs)
    return std::nullopt;
  auto rhs = tryNarrowFeeder(muxOp.getFalseValue(), sigOp, selfFeedback,
                             depth + 1);
  if (!rhs)
    return std::nullopt;

  // An empty union means the drive provably writes nothing at all. Rather
  // than emit zero intervals -- which would drop the drive and silently
  // remove logic, a failure mode this project has had to retract twice --
  // fall back to the conservative full-width answer.
  if (lhs->empty() && rhs->empty())
    return std::nullopt;

  // Union the two arms' ranges, COALESCING ranges that overlap or touch into
  // their hull. `NarrowedWrite` records only WHERE, and the caller rebuilds
  // each range by extracting it back out of the register's own result, so
  // widening a range to a hull can only move the answer toward the
  // conservative full-width one -- never past it. It stays a narrowing
  // because ranges separated by a real gap are kept apart.
  //
  // Refusing on any overlap (what this did before) was over-strict and cost
  // real promotions. OpenC910's `ct_mmu_ptw.lsu_data_flop` is written by two
  // sibling always blocks over DISJOINT slices, [58:0] and [63:59]. The
  // [58:0] one is a mux tree whose arms write [0+59] (the whole slice, from
  // `lsu_mmu_data`) and, in the cross-page-walk arms, [19+9] and [10+9] --
  // both CONTAINED in [0+59], which is not a conflict at all. Bailing to full
  // width there made the two disjoint drives look like they collide.
  SmallVector<NarrowedWrite> merged(*lhs);
  merged.append(rhs->begin(), rhs->end());
  llvm::sort(merged, [](const NarrowedWrite &a, const NarrowedWrite &b) {
    if (a.bitOffset != b.bitOffset)
      return a.bitOffset < b.bitOffset;
    return a.bitwidth < b.bitwidth;
  });

  SmallVector<NarrowedWrite> coalesced;
  for (const auto &w : merged) {
    if (!coalesced.empty()) {
      NarrowedWrite &back = coalesced.back();
      uint64_t backEnd = back.bitOffset + back.bitwidth;
      // `<=` rather than `<`: adjacent ranges merge too. Sorted by offset, so
      // `w.bitOffset >= back.bitOffset` and the hull starts at back's offset.
      if (w.bitOffset <= backEnd) {
        back.bitwidth =
            std::max(backEnd, w.bitOffset + w.bitwidth) - back.bitOffset;
        continue;
      }
    }
    coalesced.push_back(w);
  }
  return coalesced;
}

class SigPromoter {
public:
  SigPromoter(llhd::SignalOp sigOp) : sigOp(sigOp) {}

  // Start at the signal operation and traverse all alias operations to compute
  // all the intervals and sort them by ascending offset.
  LogicalResult computeIntervals() {
    SmallVector<std::pair<Operation *, Offset>> stack;

    for (auto *user : sigOp->getUsers())
      stack.emplace_back(user, Offset(0));

    while (!stack.empty()) {
      auto currAndOffset = stack.pop_back_val();
      auto *curr = currAndOffset.first;
      auto offset = currAndOffset.second;

      if (curr->getBlock() != sigOp->getBlock()) {
        // A user nested directly inside an `llhd.combinational` op can be
        // treated as if it were in the signal's own block when that op's
        // body is a single block: `llhd.combinational` regions are only
        // constrained to MinSizedRegion<1>, so they can in general contain
        // real control flow (multiple blocks joined by `cf.cond_br`, per
        // the op's own documented example) whose reaching-drive semantics
        // this interval-based algorithm does not attempt to model -- but a
        // single-block body has no such branching and, by the op's
        // documented semantics ("executes once... whenever any of the
        // values used in its body change"), always runs end-to-end
        // unconditionally, exactly like inlined straight-line code. This
        // lets signals driven inside a straight-line `llhd.combinational`
        // (the common shape for e.g. an unrolled `always_comb` loop body)
        // but probed from the enclosing module body promote instead of
        // bailing outright.
        auto combOp =
            dyn_cast_or_null<llhd::CombinationalOp>(curr->getParentOp());
        bool transparent = combOp && combOp->getBlock() == sigOp->getBlock() &&
                           combOp.getBody().hasOneBlock();
        if (!transparent) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  - User in other block, skipping...\n\n");
          return failure();
        }
      }

      auto result =
          TypeSwitch<Operation *, LogicalResult>(curr)
              .Case<llhd::ProbeOp>([&](llhd::ProbeOp probeOp) {
                auto bw = hw::getBitWidth(probeOp.getResult().getType());
                if (bw <= 0)
                  return failure();

                readIntervals.emplace_back(offset, bw, probeOp.getResult(),
                                           curr->getBlock());
                return success();
              })
              .Case<llhd::DriveOp>([&](llhd::DriveOp driveOp) {
                if (driveOp.getEnable()) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Conditional driver, skipping...\n\n");
                  return failure();
                }

                auto timeOp =
                    driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
                if (!timeOp) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  - Unknown drive delay, skipping...\n\n");
                  return failure();
                }

                auto bw = hw::getBitWidth(driveOp.getValue().getType());
                if (bw <= 0)
                  return failure();

                // A whole-signal drive fed by a `seq.firreg` whose `next`
                // operand (modulo an enclosing clock-enable self-feedback
                // mux) is an array-inject chain, or a bit-concat of
                // probe-extracts and new bits, over a probe of this same
                // signal is the per-element/per-bitfield clocked-write
                // shape described above: narrow to one interval per
                // written slice, each valued by extracting that slice back
                // out of the firreg's own RESULT (not out of `next` -- the
                // signal's future value at that position is what the
                // register will hold, i.e. its result, not the
                // combinational value feeding it).
                if (offset.isStatic() && offset.min == 0) {
                  // The narrowable shape usually sits behind a register (the
                  // drive value is a seq.firreg and its `next` carries the
                  // pass-through), but not always: a purely combinational
                  // drive can carry it directly, e.g. OpenC910's
                  // `ifu_icache_tag_wen` is driven by
                  // `concat(%new, extract(prb(self), 0))` with no register in
                  // between. Gating on the firreg missed those entirely and
                  // reported them at full width.
                  auto firregOp = driveOp.getValue().getDefiningOp<seq::FirRegOp>();
                  {
                    // Which value carries the narrowable shape:
                    //  * no register        -> the drive value itself
                    //  * `next` is the reg  -> the register never changes, so
                    //    it holds its RESET value forever and that is where
                    //    the shape lives. OpenC910's `ct_prio*` initialise an
                    //    array per element this way:
                    //        %r = seq.firreg %r ... reset %rst,
                    //             hw.array_inject (llhd.prb %sig)[k], %c
                    //    Reading `next` there just finds the register itself,
                    //    which narrows to nothing and forces full width.
                    //  * otherwise          -> `next`
                    Value feeder;
                    Value selfFeedback;
                    if (!firregOp) {
                      feeder = driveOp.getValue();
                    } else if (firregOp.getNext() == firregOp.getResult() &&
                               firregOp.getResetValue()) {
                      feeder = firregOp.getResetValue();
                      selfFeedback = firregOp.getResult();
                    } else {
                      feeder = firregOp.getNext();
                      selfFeedback = firregOp.getResult();
                    }
                    auto narrowed = tryNarrowFeeder(feeder, sigOp, selfFeedback);
                    LLVM_DEBUG({
                      if (narrowed) {
                        auto parent =
                            sigOp->getParentOfType<hw::HWModuleOp>();
                        llvm::dbgs()
                            << "  - narrowed "
                            << (!firregOp ? "combinational"
                                          : (firregOp.getNext() ==
                                                     firregOp.getResult() &&
                                             firregOp.getResetValue()
                                                 ? "reset-value"
                                                 : "next"))
                            << " drive of @"
                            << (parent ? parent.getModuleName() : "?") << "."
                            << sigOp.getName().value_or("?") << " to "
                            << narrowed->size() << " range(s)\n";
                      }
                    });
                    // A struct-typed drive can only be sliced back apart with
                    // `hw.struct_extract`, which addresses a FIELD, not a bit
                    // range -- so every narrowed range has to land exactly on
                    // one field. It normally does (each came from a
                    // struct_inject of one field), but `tryNarrowFeeder`'s mux
                    // union coalesces adjacent ranges into a hull, and a hull
                    // spanning two fields has no single field to name. Resolve
                    // the field names up front and abandon the whole narrowing
                    // if any range fails to match, rather than building an
                    // invalid `comb.extract` on a struct.
                    auto structTy = hw::type_dyn_cast<hw::StructType>(
                        driveOp.getValue().getType());
                    SmallVector<StringAttr> fieldNames;
                    if (narrowed && structTy) {
                      auto elements = structTy.getElements();
                      for (auto &write : *narrowed) {
                        uint64_t offset = hw::getBitWidth(structTy);
                        StringAttr match;
                        for (auto &element : elements) {
                          auto fieldBw = hw::getBitWidth(element.type);
                          if (fieldBw < 0)
                            break;
                          offset -= fieldBw;
                          if (offset == write.bitOffset &&
                              (uint64_t)fieldBw == write.bitwidth) {
                            match = element.name;
                            break;
                          }
                        }
                        if (!match) {
                          narrowed = std::nullopt;
                          break;
                        }
                        fieldNames.push_back(match);
                      }
                    }
                    if (narrowed) {
                      bool isArray = isa<hw::ArrayType>(driveOp.getValue().getType());
                      auto arrayTy = isArray ? cast<hw::ArrayType>(
                                                  driveOp.getValue().getType())
                                            : hw::ArrayType();
                      uint64_t elemBw =
                          isArray ? hw::getBitWidth(arrayTy.getElementType()) : 0;
                      OpBuilder b(driveOp);
                      for (auto [writeIdx, write] : llvm::enumerate(*narrowed)) {
                        Value elem;
                        if (structTy) {
                          elem = hw::StructExtractOp::create(
                              b, driveOp.getLoc(), driveOp.getValue(),
                              fieldNames[writeIdx]);
                        } else if (isArray) {
                          uint64_t index = write.bitOffset / elemBw;
                          Value idx = hw::ConstantOp::create(
                              b, driveOp.getLoc(),
                              APInt(llvm::Log2_64_Ceil(arrayTy.getNumElements()),
                                   index));
                          elem = hw::ArrayGetOp::create(b, driveOp.getLoc(),
                                                        driveOp.getValue(), idx);
                        } else {
                          elem = comb::ExtractOp::create(
                              b, driveOp.getLoc(),
                              b.getIntegerType(write.bitwidth),
                              driveOp.getValue(), write.bitOffset);
                        }
                        intervals.emplace_back(Offset(write.bitOffset),
                                               write.bitwidth, elem,
                                               curr->getBlock(),
                                               timeOp.getValueAttr());
                      }
                      return success();
                    }
                  }
                }

                intervals.emplace_back(offset, bw, driveOp.getValue(),
                                       curr->getBlock(),
                                       timeOp.getValueAttr());
                return success();
              })
              .Case<llhd::SigExtractOp>([&](llhd::SigExtractOp extractOp) {
                if (auto constOp =
                        extractOp.getLowBit().getDefiningOp<hw::ConstantOp>();
                    constOp && offset.isStatic()) {
                  for (auto *user : extractOp->getUsers())
                    stack.emplace_back(
                        user,
                        Offset(constOp.getValue().getZExtValue() + offset.min));

                  return success();
                }

                auto bw = hw::getBitWidth(
                    cast<llhd::RefType>(extractOp.getInput().getType())
                        .getNestedType());
                if (bw <= 0)
                  return failure();

                SmallVector<Value> indices(offset.dynamic);
                indices.push_back(extractOp.getLowBit());

                for (auto *user : extractOp->getUsers())
                  stack.emplace_back(
                      user, Offset(offset.min, offset.max + bw - 1, indices));

                return success();
              })
              // The three remaining ways to alias a signal. Without these a
              // signal reached through a struct field or an array element is
              // simply not promotable, which on CVA6 was 148 ops across 20
              // modules -- every one of them ordinary RTL writing one field
              // of a packed struct.
              //
              // BIT LAYOUT, MEASURED not assumed. `hw-aggregate-to-comb` is
              // the authority, because it is what actually flattens these
              // types later, and the two conventions are OPPOSITE:
              //
              //   !hw.array<2xi4> from i8   -> element 0 at bit 0 (LOW)
              //   !hw.struct<a:i2,b:i6,c:i4>
              //        from i12            -> a at bit 10, b at 4, c at 0,
              //                               i.e. the FIRST field is HIGH
              //
              // so a struct field's offset is the total width of the fields
              // AFTER it. Guessing this symmetric would have been wrong, and
              // an offset bug here is a silent miscompilation -- see the
              // array-offset note on tryNarrowArrayInjectChain above, which
              // is exactly that bug, found by a d2 equivalence check.
              .Case<llhd::SigStructExtractOp>(
                  [&](llhd::SigStructExtractOp structOp) {
                    if (!offset.isStatic())
                      return failure();
                    auto structTy = hw::type_cast<hw::StructType>(
                        cast<llhd::RefType>(structOp.getInput().getType())
                            .getNestedType());
                    auto index = structTy.getFieldIndex(structOp.getFieldAttr());
                    if (!index)
                      return failure();
                    auto elements = structTy.getElements();
                    uint64_t below = 0;
                    for (size_t i = *index + 1, e = elements.size(); i < e; ++i) {
                      auto bw = hw::getBitWidth(elements[i].type);
                      if (bw < 0)
                        return failure();
                      below += bw;
                    }
                    for (auto *user : structOp->getUsers())
                      stack.emplace_back(user, Offset(offset.min + below));
                    return success();
                  })
              .Case<llhd::SigArrayGetOp>([&](llhd::SigArrayGetOp getOp) {
                auto constIdx = getOp.getIndex().getDefiningOp<hw::ConstantOp>();
                if (!constIdx || !offset.isStatic())
                  return failure();
                auto arrayTy = hw::type_cast<hw::ArrayType>(
                    cast<llhd::RefType>(getOp.getInput().getType())
                        .getNestedType());
                auto elemBw = hw::getBitWidth(arrayTy.getElementType());
                if (elemBw <= 0)
                  return failure();
                uint64_t elemOffset =
                    constIdx.getValue().getZExtValue() * (uint64_t)elemBw;
                for (auto *user : getOp->getUsers())
                  stack.emplace_back(user, Offset(offset.min + elemOffset));
                return success();
              })
              .Case<llhd::SigArraySliceOp>([&](llhd::SigArraySliceOp sliceOp) {
                auto constIdx =
                    sliceOp.getLowIndex().getDefiningOp<hw::ConstantOp>();
                if (!constIdx || !offset.isStatic())
                  return failure();
                auto arrayTy = hw::type_cast<hw::ArrayType>(
                    cast<llhd::RefType>(sliceOp.getInput().getType())
                        .getNestedType());
                auto elemBw = hw::getBitWidth(arrayTy.getElementType());
                if (elemBw <= 0)
                  return failure();
                uint64_t elemOffset =
                    constIdx.getValue().getZExtValue() * (uint64_t)elemBw;
                for (auto *user : sliceOp->getUsers())
                  stack.emplace_back(user, Offset(offset.min + elemOffset));
                return success();
              })
              .Default([](auto *op) {
                LLVM_DEBUG(llvm::dbgs() << "  - User that is not a probe or "
                                           "drive, skipping...\n    "
                                        << *op << "\n\n");
                return failure();
              });

      if (failed(result))
        return failure();

      toDelete.push_back(curr);
    }

    llvm::sort(intervals, [](const Interval &a, const Interval &b) {
      return a.low.min < b.low.min;
    });

    LLVM_DEBUG({
      llvm::dbgs() << "  - Detected intervals:\n";
      dumpIntervals(llvm::dbgs(), 4);
    });

    return success();
  }

#ifndef NDEBUG

  /// Print the list of intervals in a readable format for debugging.
  void dumpIntervals(llvm::raw_ostream &os, unsigned indent = 0) {
    os << llvm::indent(indent) << "[\n";
    for (const auto &interval : intervals) {
      os << llvm::indent(indent + 2) << "<from [" << interval.low.min << ", "
         << interval.low.max << "]\n";
      os << llvm::indent(indent + 3) << "width " << interval.bitwidth << "\n";

      for (auto idx : interval.low.dynamic)
        os << llvm::indent(indent + 3) << idx << "\n";

      os << llvm::indent(indent + 3) << "value: " << interval.value << "\n";
      os << llvm::indent(indent + 3) << "delay: " << interval.delay << "\n";
      os << llvm::indent(indent + 2) << ">,\n";
    }
    os << llvm::indent(indent) << "]\n";
  }

#endif

  /// Check if we can promote the entire signal according to the current
  /// limitations of the pass.
  bool isPromotable() {
    for (unsigned i = 0; i < intervals.size(); ++i) {
      if (i >= intervals.size() - 1)
        break;

      if (intervals[i].low.max + intervals[i].bitwidth - 1 >
          intervals[i + 1].low.min) {
        LLVM_DEBUG({
          llvm::dbgs() << "  - Potentially overlapping drives, skipping...\n\n";
        });
        return false;
      }
    }

    return true;
  }

  /// Promote the signal. This builds the necessary operations, replaces the
  /// values, and removes the signal and signal value handling operations.
  void promote() {
    auto bw = hw::getBitWidth(sigOp.getInit().getType());
    assert(bw > 0 && "bw must be known and non-zero");

    // Normally every interval lives in sigOp's own block and the whole
    // reconstruction can be built right at sigOp. The single-block
    // `llhd.combinational` transparency carve-out above means some
    // interval's originating probe/drive may instead live inside that op's
    // body -- in that case the reconstruction must be built there instead
    // (the only point that dominates values defined inside it), and any
    // read whose ORIGINAL probe was outside that block (i.e. back at
    // sigOp's own block) has to have its resolved value threaded back out
    // through the combinational op's results, since a value defined in a
    // child region cannot directly satisfy a use in the parent.
    Block *sigBlock = sigOp->getBlock();
    Block *foreignBlock = nullptr;
    for (auto &interval : intervals)
      if (interval.block != sigBlock)
        foreignBlock = interval.block;
    for (auto &interval : readIntervals)
      if (interval.block != sigBlock)
        foreignBlock = interval.block;
    assert((!foreignBlock ||
            llvm::all_of(intervals,
                         [&](auto &i) {
                           return i.block == sigBlock ||
                                  i.block == foreignBlock;
                         })) &&
           "at most one foreign block supported");

    // `outerBuilder` sits at sigOp's own position, which -- because a value
    // defined in a parent block is visible everywhere in a nested region --
    // dominates every point inside `foreignBlock` regardless of where in
    // that block a given use sits. `preVal` (the signal's value before any
    // of this reconstruction's writes) is built here, and is what any read
    // whose ORIGINAL probe lived inside `foreignBlock` must resolve to:
    // such a read is, structurally, always the seed mem2reg's own rewrite
    // chain was built from (e.g. `%4 = llhd.prb %sig` feeding the first
    // `hw.array_inject` in the examples above) -- i.e. the value as of
    // entry to the block, before this pass's own reconstructed writes are
    // applied -- never some partially-written intermediate state, since
    // mem2reg does not re-probe the raw signal mid-chain. Resolving it to
    // the fully-POST-write `val` instead (as a same-block read normally
    // would) would make that value depend on its own seed, a genuine
    // circular/non-dominating definition -- exactly the dominance failure
    // this split avoids.
    OpBuilder outerBuilder(sigOp);
    Location loc = sigOp->getLoc();
    auto type = outerBuilder.getIntegerType(bw);
    Value preVal =
        outerBuilder.createOrFold<hw::BitcastOp>(loc, type, sigOp.getInit());

    // `builder` builds the write-reconstruction chain and, when there is a
    // foreign block, must live inside it (the only point that dominates
    // interval values defined there); with no foreign block this is exactly
    // the original single-block behavior, unchanged.
    OpBuilder builder = outerBuilder;
    if (foreignBlock)
      builder.setInsertionPoint(foreignBlock->getTerminator());
    Value val = preVal;

    // Handle the writes by starting with the signal init value and injecting
    // the written values at the right offsets.
    for (auto interval : intervals) {
      Value invMask = hw::ConstantOp::create(
          builder, loc, APInt::getAllOnes(interval.bitwidth));

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = hw::ConstantOp::create(
            builder, loc, APInt::getZero(bw - interval.bitwidth));
        invMask = builder.createOrFold<comb::ConcatOp>(loc, pad, invMask);
      }

      Value amt = buildDynamicIndex(builder, loc, interval.low.min,
                                    interval.low.dynamic, bw);
      invMask = builder.createOrFold<comb::ShlOp>(loc, invMask, amt);
      Value allOnes =
          hw::ConstantOp::create(builder, loc, APInt::getAllOnes(bw));
      Value mask = builder.createOrFold<comb::XorOp>(loc, invMask, allOnes);
      val = builder.createOrFold<comb::AndOp>(loc, val, mask);

      Value assignVal = builder.createOrFold<hw::BitcastOp>(
          loc, builder.getIntegerType(interval.bitwidth), interval.value);

      if (uint64_t(bw) > interval.bitwidth) {
        Value pad = hw::ConstantOp::create(
            builder, loc, APInt::getZero(bw - interval.bitwidth));
        assignVal = builder.createOrFold<comb::ConcatOp>(loc, pad, assignVal);
      }

      assignVal = builder.createOrFold<comb::ShlOp>(loc, assignVal, amt);
      if (!isImmediate(interval.delay))
        assignVal =
            builder.createOrFold<llhd::DelayOp>(loc, assignVal, interval.delay);
      val = builder.createOrFold<comb::OrOp>(loc, assignVal, val);
    }

    // Handle the reads by extracting right number of bits at the right offset.
    // Reads whose original probe lived outside the block the reconstruction
    // was built in (i.e. back in sigOp's own block, while `val` was built
    // inside a foreign transparent `llhd.combinational`) can't be wired up
    // via replaceAllUsesWith directly -- collected here and threaded out
    // through the combinational op's results afterward instead.
    SmallVector<std::pair<Value, Value>> escaping;
    for (auto interval : readIntervals) {
      // A read whose original probe lived inside the foreign block resolves
      // against the PRE-write value, built with the outer builder (see the
      // comment above `preVal`); every other read -- either the ordinary
      // single-block case, or one that must escape a foreign block --
      // resolves against the fully-reconstructed POST-write `val`.
      bool isForeignSeed = foreignBlock && interval.block == foreignBlock;
      OpBuilder &b = isForeignSeed ? outerBuilder : builder;
      Value base = isForeignSeed ? preVal : val;

      Value read;
      if (interval.low.isStatic()) {
        read = b.createOrFold<comb::ExtractOp>(
            loc, b.getIntegerType(interval.bitwidth), base, interval.low.min);
      } else {
        Value amt = buildDynamicIndex(b, loc, interval.low.min,
                                      interval.low.dynamic, bw);
        read = b.createOrFold<comb::ShrUOp>(loc, base, amt);
        read = b.createOrFold<comb::ExtractOp>(
            loc, b.getIntegerType(interval.bitwidth), read, 0);
      }
      read = b.createOrFold<hw::BitcastOp>(loc, interval.value.getType(), read);

      if (read == interval.value)
        continue;
      if (foreignBlock && interval.block != foreignBlock)
        escaping.emplace_back(interval.value, read);
      else
        interval.value.replaceAllUsesWith(read);
    }

    // Thread any escaping values (resolved inside the foreign
    // `llhd.combinational`, but originally probed from sigOp's own block)
    // out through that op's results: MLIR op result lists are fixed at
    // creation, so this rebuilds the op with the additional result types,
    // moves its body over unchanged, extends its `llhd.yield` with the
    // escaping values, and redirects both the op's pre-existing result uses
    // and each escaping probe's uses to the appropriate result of the new op.
    if (!escaping.empty()) {
      auto oldCombOp = cast<llhd::CombinationalOp>(foreignBlock->getParentOp());
      SmallVector<Type> resultTypes(oldCombOp.getResultTypes());
      for (auto &[oldValue, newValue] : escaping)
        resultTypes.push_back(newValue.getType());

      OpBuilder combBuilder(oldCombOp);
      auto newCombOp = llhd::CombinationalOp::create(combBuilder,
                                                     oldCombOp.getLoc(),
                                                     resultTypes);
      newCombOp.getBody().takeBody(oldCombOp.getBody());

      auto yieldOp =
          cast<llhd::YieldOp>(newCombOp.getBody().front().getTerminator());
      SmallVector<Value> yieldOperands(yieldOp.getOperands());
      for (auto &[oldValue, newValue] : escaping)
        yieldOperands.push_back(newValue);
      yieldOp->setOperands(yieldOperands);

      for (auto [oldResult, newResult] :
           llvm::zip(oldCombOp.getResults(), newCombOp.getResults()))
        oldResult.replaceAllUsesWith(newResult);
      for (auto [i, pair] : llvm::enumerate(escaping))
        pair.first.replaceAllUsesWith(
            newCombOp.getResult(oldCombOp.getNumResults() + i));

      oldCombOp->erase();
    }

    // Delete all operations operating on signal values.
    for (auto *op : llvm::reverse(toDelete))
      op->erase();

    sigOp->erase();
  }

private:
  /// Given a static offset and a list of dynamic offset values, materialize an
  /// SSA value that adds all these offsets together and is an integer with the
  /// given 'width'.
  Value buildDynamicIndex(OpBuilder &builder, Location loc,
                          uint64_t constOffset, ArrayRef<Value> indices,
                          uint64_t width) {
    Value index = hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(width), constOffset);

    for (auto idx : indices) {
      auto bw = hw::getBitWidth(idx.getType());
      Value pad =
          hw::ConstantOp::create(builder, loc, APInt::getZero(width - bw));
      idx = builder.createOrFold<comb::ConcatOp>(loc, pad, idx);
      index = builder.createOrFold<comb::AddOp>(loc, index, idx);
    }

    return index;
  }

  bool isImmediate(llhd::TimeAttr attr) const {
    return attr.getTime() == 0 && attr.getDelta() == 0 &&
           attr.getEpsilon() == 1;
  }

  // The signal to be promoted.
  llhd::SignalOp sigOp;
  // Intervals written to.
  SmallVector<Interval> intervals;
  // Intervals read from.
  SmallVector<Interval> readIntervals;
  // Operations to delete after promotion is done.
  SmallVector<Operation *> toDelete;
};

struct Sig2RegPass : public circt::llhd::impl::Sig2RegBase<Sig2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Sig2RegPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== Sig2Reg in module " << moduleOp.getSymName()
                          << "\n\n");

  for (auto sigOp :
       llvm::make_early_inc_range(moduleOp.getOps<llhd::SignalOp>())) {
    // If the signal is only driven, but never read or used otherwise, remove
    // it.
    if (llvm::all_of(sigOp->getUses(), [](auto &use) {
          return isa<llhd::DriveOp>(use.getOwner()) &&
                 use.getOperandNumber() == 0;
        })) {
      LLVM_DEBUG(llvm::dbgs() << "  - Removing drive-only signal "
                              << sigOp.getName() << "\n");
      for (auto *user : llvm::make_early_inc_range(sigOp->getUsers()))
        user->erase();
      sigOp.erase();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "  - Attempting to promote signal "
                            << sigOp.getName() << "\n");
    SigPromoter promoter(sigOp);
    if (failed(promoter.computeIntervals()) || !promoter.isPromotable())
      continue;

    promoter.promote();
    LLVM_DEBUG(llvm::dbgs() << "  - Successfully promoted!\n\n");
  }

  LLVM_DEBUG({
    if (moduleOp.getOps<llhd::SignalOp>().empty())
      llvm::dbgs() << "  Successfully promoted all signals in module!\n";
  });

  LLVM_DEBUG(llvm::dbgs() << "\n");
}
