// RUN: circt-opt --llhd-remove-control-flow %s | FileCheck %s

// CHECK-LABEL: @Basic
hw.module @Basic(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP0:%.+]] = comb.icmp eq %a, %b
    %0 = comb.icmp eq %a, %b : i42
    // CHECK-NEXT: [[TMP1:%.+]] = comb.add %a, %b
    %1 = comb.add %a, %b : i42
    // CHECK-NOT: cf.br
    cf.br ^bb1
  ^bb1:
    // CHECK-NEXT: [[TMP2:%.+]] = comb.sub %a, %b
    %2 = comb.sub %a, %b : i42
    // CHECK-NOT: cf.cond_br
    cf.cond_br %0, ^bb2(%1 : i42), ^bb3(%2 : i42)
  ^bb2(%3: i42):
    // CHECK-NEXT: [[TMP4:%.+]] = comb.mul [[TMP1]], [[TMP2]]
    %4 = comb.mul %3, %2 : i42
    // CHECK-NEXT: [[TMP5:%.+]] = comb.icmp eq [[TMP4]], %b
    %5 = comb.icmp eq %4, %b : i42
    // CHECK-NOT: cf.cond_br
    cf.cond_br %5, ^bb3(%4 : i42), ^bb4(%3 : i42)
  ^bb3(%6: i42):
    // The join is lowered by branch structure: bb3 is entered from bb0's
    // false arm or, via bb2, from bb0's true arm. The bb2 branch (on TMP5)
    // does not select between the two values because its other arm never
    // reaches bb3, so it does not appear in the mux.
    // CHECK-NEXT: [[TMP6:%.+]] = comb.mux [[TMP0]], [[TMP4]], [[TMP2]]
    // CHECK-NEXT: [[TMP7:%.+]] = comb.xor [[TMP1]], [[TMP6]]
    %7 = comb.xor %1, %6 : i42
    // CHECK-NOT: cf.br
    cf.br ^bb4(%7 : i42)
  ^bb4(%8: i42):
    // CHECK-NEXT: [[TMP8A:%.+]] = comb.mux [[TMP5]], [[TMP7]], [[TMP1]]
    // CHECK-NEXT: [[TMP8:%.+]] = comb.mux [[TMP0]], [[TMP8A]], [[TMP7]]
    // CHECK-NEXT: llhd.yield [[TMP8]]
    llhd.yield %8 : i42
  }
}

// A value assigned before a nested if that only assigns OTHER variables
// arrives at the join unchanged along every edge under that if. Lowering by
// per-edge path predicates rebuilt it as v&~g | v&g&r | v&g&~r, which made
// it structurally depend on g and r (and closed a false combinational loop
// whenever g was computed from it). Structured lowering yields `v` outright.
// CHECK-LABEL: @NoDependenceOnBranchesThatDoNotChangeValue
hw.module @NoDependenceOnBranchesThatDoNotChangeValue(in %v: i1, in %g: i1, in %r: i1, in %q: i2) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i1, i2 {
    %true = hw.constant true
    %false = hw.constant false
    %c1_i2 = hw.constant 1 : i2
    %c-1_i2 = hw.constant -1 : i2
    cf.cond_br %v, ^bb1, ^bb4(%false, %q : i1, i2)
  ^bb1:
    cf.cond_br %g, ^bb2, ^bb4(%true, %q : i1, i2)
  ^bb2:
    cf.cond_br %r, ^bb3(%c1_i2 : i2), ^bb3(%c-1_i2 : i2)
  ^bb3(%inc: i2):
    // CHECK: [[INC:%.+]] = comb.mux %r, %c1_i2, %c-1_i2
    // CHECK-NEXT: [[SUM:%.+]] = comb.add %q, [[INC]]
    %sum = comb.add %q, %inc : i2
    cf.br ^bb4(%true, %sum : i1, i2)
  ^bb4(%req: i1, %cnt: i2):
    // CHECK-NEXT: [[CNT1:%.+]] = comb.mux %g, [[SUM]], %q
    // CHECK-NEXT: [[CNT:%.+]] = comb.mux %v, [[CNT1]], %q
    // CHECK-NEXT: llhd.yield %v, [[CNT]]
    // CHECK-NOT: comb.and
    // CHECK-NOT: comb.or
    llhd.yield %req, %cnt : i1, i2
  }
}

// A switch-like chain of conditional branches into one join becomes a
// priority mux chain on the raw branch conditions, not on accumulated path
// predicates.
// CHECK-LABEL: @BranchChainIsPriorityMuxChain
hw.module @BranchChainIsPriorityMuxChain(in %a: i1, in %b: i1, in %x: i8, in %y: i8, in %z: i8) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i8 {
    cf.cond_br %a, ^bb2(%x : i8), ^bb1
  ^bb1:
    cf.cond_br %b, ^bb2(%y : i8), ^bb2(%z : i8)
  ^bb2(%0: i8):
    // CHECK-NEXT: [[T1:%.+]] = comb.mux %b, %y, %z
    // CHECK-NEXT: [[T0:%.+]] = comb.mux %a, %x, [[T1]]
    // CHECK-NEXT: llhd.yield [[T0]]
    llhd.yield %0 : i8
  }
}

// The enable folded into a drive is built the same way: a branch both of
// whose arms lead to the drive's block does not appear in it.
// CHECK-LABEL: @DriveEnableIgnoresReconvergentBranches
hw.module @DriveEnableIgnoresReconvergentBranches(in %a: i1, in %b: i1, in %d: i8) {
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  // CHECK: [[T:%.+]] = llhd.constant_time
  %t = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    cf.cond_br %a, ^bb1, ^bb4
  ^bb1:
    cf.cond_br %b, ^bb2, ^bb3
  ^bb2:
    cf.br ^bb3
  ^bb3:
    // CHECK-NEXT: llhd.drv %sig, %d after [[T]] if %a
    // CHECK-NEXT: llhd.yield
    llhd.drv %sig, %d after %t : i8
    cf.br ^bb4
  ^bb4:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipWhenSideEffectsPresent
hw.module @SkipWhenSideEffectsPresent() {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK: cf.br
    cf.br ^bb1
  ^bb1:
    func.call @someFunc() : () -> ()
    llhd.yield
  }
}

// CHECK-LABEL: @SkipWhenLoopsPresent
hw.module @SkipWhenLoopsPresent(in %a: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK: cf.br
    cf.br ^bb1
  ^bb1:
    // CHECK: cf.cond_br
    cf.cond_br %a, ^bb1, ^bb2
  ^bb2:
    llhd.yield
  }
}

// CHECK-LABEL: @IgnoreValuesComingFromUnreachableBlock
hw.module @IgnoreValuesComingFromUnreachableBlock(in %a: i42, in %b: i42) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: llhd.yield %a
    cf.br ^bb1(%a : i42)
  ^bb1(%0: i42):
    llhd.yield %0 : i42
  ^bb2:
    cf.br ^bb1(%b : i42)
  }
}

// CHECK-LABEL: @MultipleYields
hw.module @MultipleYields(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP:%.+]] = comb.mux %c, %a, %b
    // CHECK-NEXT: llhd.yield [[TMP]]
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    llhd.yield %a : i42
  ^bb2:
    llhd.yield %b : i42
  }
}

// CHECK-LABEL: @HandleCondBranchToSameBlock
hw.module @HandleCondBranchToSameBlock(in %a: i42, in %b: i42, in %c: i1) {
  // CHECK-NEXT: llhd.combinational
  llhd.combinational -> i42 {
    // CHECK-NEXT: [[TMP:%.+]] = comb.mux %c, %a, %b
    // CHECK-NEXT: llhd.yield [[TMP]]
    cf.cond_br %c, ^bb1(%a : i42), ^bb1(%b : i42)
  ^bb1(%0: i42):
    llhd.yield %0 : i42
  }
}

func.func private @someFunc()
