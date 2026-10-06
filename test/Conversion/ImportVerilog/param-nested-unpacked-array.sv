// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-hw %s | FileCheck %s --check-prefix=HW
// REQUIRES: slang

// A 2-D unpacked-array parameter read with a loop index inside a function
// used to fail with "unknown name" / "no rvalue generated for Parameter":
// the constant materializer only handled one level of integer elements.

// CHECK-LABEL: moore.module private @Lut(
// CHECK: moore.array_create {{.+}} -> uarray<2 x i32>
// CHECK: moore.array_create {{.+}} -> uarray<2 x i32>
// CHECK: moore.array_create {{.+}} : !moore.uarray<2 x i32>, !moore.uarray<2 x i32> -> uarray<2 x uarray<2 x i32>>
module Lut #(
  parameter int unsigned N = 1,
  parameter int unsigned Map [N-1:0][0:1] = '{default: {32'd0, 32'd0}}
) (output logic [7:0] m);
  function automatic logic [7:0] build();
    logic [7:0] ret = '0;
    for (int unsigned k = 0; k < N; ++k)
      ret[Map[k][0]] = Map[k][1];
    return ret;
  endfunction
  localparam logic [7:0] Bits = build();
  assign m = Bits;
endmodule

// Map entries {2,3} and {5,1} set bits 2 and 5 of the specialised Lut.
// HW-LABEL: hw.module private @Lut(
// HW: hw.constant 36 : i8
module Top (output logic [7:0] m);
  Lut #(.N(2), .Map('{'{32'd2, 32'd3}, '{32'd5, 32'd1}})) u (.m(m));
endmodule
