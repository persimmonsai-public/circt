// RUN: circt-opt %s --hw-aggregate-to-comb | FileCheck %s

// Packed unions flatten to integers. A member as wide as the union covers
// it; a narrower member sits at offset-from-MSB, as ExportVerilog pads it.

// CHECK-LABEL: @union_in_struct
// CHECK: [[U:%.+]] = hw.bitcast %u : (!hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>) -> i8
// CHECK: comb.concat %v, [[U]] : i1, i8
// CHECK: [[XA:%.+]] = comb.extract [[U]] from 4 : (i8) -> i4
// CHECK: hw.output {{.+}}, [[XA]]
hw.module @union_in_struct(in %u : !hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>, in %v : i1,
                           out o : !hw.struct<v: i1, d: !hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>>, out a : i4) {
  %s = hw.struct_create (%v, %u) : !hw.struct<v: i1, d: !hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>>
  %x = hw.union_extract %u["x"] : !hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>
  %a = hw.struct_extract %x["a"] : !hw.struct<a: i4, b: i4>
  hw.output %s, %a : !hw.struct<v: i1, d: !hw.union<x: !hw.struct<a: i4, b: i4>, y: i8>>, i4
}

// CHECK-LABEL: @narrow_member
// The 3-bit member at offset 0 is the top 3 bits of the 8-bit union.
// CHECK: [[Z:%.+]] = hw.constant 0 : i5
// CHECK: comb.concat %n, [[Z]] : i3, i5
// CHECK: comb.extract %{{.+}} from 5 : (i8) -> i3
hw.module @narrow_member(in %n : i3, out u : !hw.union<n: i3, w: i8>, out m : i3) {
  %c = hw.union_create "n", %n : !hw.union<n: i3, w: i8>
  %e = hw.union_extract %c["n"] : !hw.union<n: i3, w: i8>
  hw.output %c, %e : !hw.union<n: i3, w: i8>, i3
}
