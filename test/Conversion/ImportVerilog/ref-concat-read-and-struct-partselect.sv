// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

// A compound assignment to a concatenation reads the concatenation back:
// `{x, y} &= m` used to leave a moore.concat_ref that MooreToCore could not
// legalize.
// CHECK-LABEL: hw.module @CompoundConcat(
// CHECK-DAG: [[LOW:%.+]] = comb.extract %b from 0 : (i10) -> i2
// CHECK-DAG: [[Y:%.+]] = comb.concat %c0_i8, [[LOW]] : i8, i2
// CHECK: hw.output %c0_i10, [[Y]] : i10, i10
module CompoundConcat(input logic [9:0] a, b, output logic [9:0] x, y);
  always_comb begin
    x = a; y = b;
    {x, y} &= 20'h3;
  end
endmodule

// A dynamic part-select write into a packed struct (legal on any packed
// type) used to leave a moore.dyn_extract_ref on a struct reference.
// CHECK-LABEL: hw.module @StructPartSelect(
// CHECK-DAG: [[D0:%.+]] = hw.array_get %d[%false]
// CHECK-DAG: [[D1:%.+]] = hw.array_get %d[%true]
// CHECK-DAG: comb.concat [[D0]], %c0_i128 : i64, i128
// CHECK-DAG: comb.concat %c0_i64, [[D1]], %c0_i64 : i64, i64, i64
// CHECK: hw.struct_inject {{%.+}}["c"], %last
module StructPartSelect(input logic [1:0][63:0] d, input logic [63:0] last, output logic [191:0] o);
  typedef struct packed { logic [63:0] a; logic [63:0] b; logic [63:0] c; } desc_t;
  desc_t cur;
  always_comb begin
    cur = '0;
    for (int i = 0; i < 2; i++)
      cur[191 - i*64 -: 64] = d[i];
    cur[63:0] = last;
  end
  assign o = cur;
endmodule
