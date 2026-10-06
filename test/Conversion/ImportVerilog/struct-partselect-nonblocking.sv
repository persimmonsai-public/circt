// RUN: not circt-verilog --ir-hw %s 2>&1 | FileCheck %s
// REQUIRES: slang

// A dynamic part-select write into a packed struct is lowered as a
// read-modify-write of the whole struct only for blocking assignments. For
// non-blocking ones that would read the struct's current value, so of the
// two writes below the second would silently undo the first. They are left
// unlowered and remain an error rather than being miscompiled.
// CHECK: error: failed to legalize operation 'moore.dyn_extract_ref'
module NonBlockingParts(input logic clk, input logic [1:0] i, j,
                        input logic [7:0] a, b, output logic [31:0] o);
  typedef struct packed { logic [15:0] hi; logic [15:0] lo; } s_t;
  s_t s;
  always_ff @(posedge clk) begin
    s[i*8 +: 8] <= a;
    s[j*8 +: 8] <= b;
  end
  assign o = s;
endmodule
