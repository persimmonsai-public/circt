//===- HWAggregateToComb.cpp - HW aggregate to comb -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWAGGREGATETOCOMB
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

// Lower hw.array_create and hw.array_concat to comb.concat.
template <typename OpTy>
struct HWArrayCreateLikeOpConversion : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInputs());
    return success();
  }
};

struct HWAggregateConstantOpConversion
    : OpConversionPattern<hw::AggregateConstantOp> {
  using OpConversionPattern<hw::AggregateConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower to concat.
    APInt intVal;
    if (failed(hw::aggregateAttrToAPInt(op.getType(), adaptor.getFieldsAttr(),
                                        intVal)))
      return failure();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, intVal);
    return success();
  }
};

struct HWArrayGetOpConversion : OpConversionPattern<hw::ArrayGetOp> {
  HWArrayGetOpConversion(TypeConverter &typeConverter, MLIRContext *ctx,
                         unsigned decodedMin)
      : OpConversionPattern<hw::ArrayGetOp>(typeConverter, ctx),
        decodedMin(decodedMin) {}
  unsigned decodedMin;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> results;
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    auto lowered = adaptor.getInput();
    auto index = adaptor.getIndex();
    APInt constantIndex;
    if (matchPattern(index, m_ConstantInt(&constantIndex))) {
      int64_t maxIndex = std::numeric_limits<int32_t>::max() / elemWidth;
      if (constantIndex.isSingleWord() &&
          constantIndex.getZExtValue() <= static_cast<uint64_t>(maxIndex)) {
        rewriter.replaceOpWithNewOp<comb::ExtractOp>(
            op, lowered, constantIndex.getZExtValue() * elemWidth, elemWidth);
        return success();
      }
    }

    for (size_t i = 0; i < numElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, elemWidth));

    // Decoded one-hot select for wide arrays: OR_i (replicate(index == i)
    // & element_i). The last element also covers out-of-bounds indices
    // (index >= N-1), matching the mux tree's documented semantics.
    auto loc = op.getLoc();
    unsigned idxWidth = index.getType().getIntOrFloatBitWidth();
    if (decodedMin > 0 && numElements >= decodedMin && idxWidth > 0 &&
        idxWidth < 64) {
      uint64_t indexable = 1ull << idxWidth;
      auto elemTy = rewriter.getIntegerType(elemWidth);
      SmallVector<Value> terms;
      for (size_t i = 0; i < numElements; ++i) {
        if (i >= indexable)
          break; // unreachable elements
        Value ci = hw::ConstantOp::create(rewriter, loc, APInt(idxWidth, i));
        bool last = i + 1 == numElements;
        Value sel = comb::ICmpOp::create(
            rewriter, loc,
            last && indexable > numElements ? comb::ICmpPredicate::uge
                                            : comb::ICmpPredicate::eq,
            index, ci, /*twoState=*/true);
        Value mask = elemWidth == 1
                         ? sel
                         : comb::ReplicateOp::create(rewriter, loc, elemTy, sel);
        terms.push_back(
            comb::AndOp::create(rewriter, loc, mask, results[i], true));
      }
      Value decoded = terms.size() == 1
                          ? terms.front()
                          : comb::OrOp::create(rewriter, loc, terms, true);
      rewriter.replaceOp(op, decoded);
      return success();
    }

    SmallVector<Value> bits;
    comb::extractBits(rewriter, index, bits);
    auto result = comb::constructMuxTree(rewriter, op.getLoc(), bits, results,
                                         results.back());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWArraySliceOpConversion : OpConversionPattern<hw::ArraySliceOp> {
  using OpConversionPattern<hw::ArraySliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> results;
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");
    auto resultArrayType = cast<hw::ArrayType>(op.getResult().getType());
    auto resultNumElements = resultArrayType.getNumElements();

    auto lowered = adaptor.getInput();
    auto index = adaptor.getLowIndex();
    APInt constantIndex;
    if (matchPattern(index, m_ConstantInt(&constantIndex))) {
      int64_t maxIndex = std::numeric_limits<int32_t>::max() / elemWidth;
      if (constantIndex.isSingleWord() &&
          constantIndex.getZExtValue() <= static_cast<uint64_t>(maxIndex)) {
        rewriter.replaceOpWithNewOp<comb::ExtractOp>(
            op, lowered, constantIndex.getZExtValue() * elemWidth,
            resultNumElements * elemWidth);
        return success();
      }
    }

    for (size_t i = 0; i <= numElements - resultNumElements; ++i)
      results.push_back(rewriter.createOrFold<comb::ExtractOp>(
          op.getLoc(), lowered, i * elemWidth, resultNumElements * elemWidth));

    SmallVector<Value> bits;
    comb::extractBits(rewriter, index, bits);
    auto result = comb::constructMuxTree(rewriter, op.getLoc(), bits, results,
                                         results.back());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct HWArrayInjectOpConversion : OpConversionPattern<hw::ArrayInjectOp> {
  using OpConversionPattern<hw::ArrayInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = cast<hw::ArrayType>(op.getInput().getType());
    auto elemType = arrayType.getElementType();
    auto numElements = arrayType.getNumElements();
    auto elemWidth = hw::getBitWidth(elemType);
    if (elemWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown element width");

    Location loc = op.getLoc();

    // Extract all elements from the input array
    SmallVector<Value> originalElements;
    auto inputArray = adaptor.getInput();
    for (size_t i = 0; i < numElements; ++i) {
      originalElements.push_back(rewriter.createOrFold<comb::ExtractOp>(
          loc, inputArray, i * elemWidth, elemWidth));
    }

    // Per-element select: out[i] = (index == i) ? newVal : old[i].
    // The previous implementation materialized an N x N "row per possible
    // inject index" matrix and selected a row with array_get -- O(N^2)
    // intermediate bits, which both blows quadratically past MLIR's
    // 16,777,215-bit IntegerType cap on real memories (CVA6 cv64a6's
    // wt_dcache_mem holds an array<256 x array<8xi44>> whose matrix would
    // be 23M bits) and wastes ABC's time even when it fits. The linear
    // form is N compares + N muxes.
    if (numElements == 1) {
      // Single-element array: the (zero-width) index can only name
      // element 0, so injection always replaces the whole array.
      rewriter.replaceOp(op, adaptor.getElement());
      return success();
    }

    Value index = adaptor.getIndex();
    SmallVector<Value> newElements;
    newElements.reserve(numElements);
    // comb.concat places its first operand at the MSBs; element N-1
    // occupies the MSBs of the lowered array, matching the extraction
    // convention above (element i lives at [i*elemWidth, (i+1)*elemWidth)).
    for (int i = numElements - 1; i >= 0; --i) {
      Value idxConst = hw::ConstantOp::create(
          rewriter, loc,
          APInt(cast<IntegerType>(index.getType()).getWidth(), i));
      Value isInject = comb::ICmpOp::create(
          rewriter, loc, comb::ICmpPredicate::eq, index, idxConst);
      newElements.push_back(comb::MuxOp::create(rewriter, loc, isInject,
                                                adaptor.getElement(),
                                                originalElements[i]));
    }
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, newElements);
    return success();
  }
};

struct HWStructCreateOpConversion : OpConversionPattern<hw::StructCreateOp> {
  using OpConversionPattern<hw::StructCreateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower struct_create to comb.concat. The first field occupies the MSBs, so
    // we concatenate fields in order (comb.concat places first operand at MSB).
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInput());
    return success();
  }
};

struct HWStructExtractOpConversion : OpConversionPattern<hw::StructExtractOp> {
  using OpConversionPattern<hw::StructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structType = cast<hw::StructType>(op.getInput().getType());
    auto fieldIndex = op.getFieldIndex();
    auto elements = structType.getElements();

    int64_t totalBitWidth = hw::getBitWidth(structType);
    if (totalBitWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown struct width");

    // Compute the bit offset from the MSB by summing the widths of all
    // preceding fields. The first field occupies the MSBs.
    int64_t consumedBits = 0;
    for (size_t i = 0; i < fieldIndex; ++i) {
      int64_t fieldWidth = hw::getBitWidth(elements[i].type);
      assert(fieldWidth >= 0 &&
             "must be failed before if field width is unknown");
      consumedBits += fieldWidth;
    }

    int64_t fieldWidth = hw::getBitWidth(elements[fieldIndex].type);
    assert(fieldWidth >= 0 &&
           "must be failed before if field width is unknown");

    // Extract the field using comb.extract. Offset is from LSB.
    int64_t bitOffset = totalBitWidth - consumedBits - fieldWidth;
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getInput(),
                                                 bitOffset, fieldWidth);
    return success();
  }
};

/// Lower hw.struct_inject to comb ops: replace the field's bit range in the
/// flattened input with the new value. Same layout convention as
/// HWStructExtractOpConversion above: the FIRST field occupies the MSBs.
///
/// Without this pattern struct_inject was not even marked illegal, so it
/// survived the pass and kept its hw.aggregate_constant operands alive --
/// on CVA6 that left 608 injects and 75 aggregate constants after
/// conversion, and every residual module carrying one was rejected as
/// unmappable to AIGER. (OpenC910 never noticed: its RTL style produces
/// array_injects, which were handled.)
struct HWStructInjectOpConversion : OpConversionPattern<hw::StructInjectOp> {
  using OpConversionPattern<hw::StructInjectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::StructInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structType = cast<hw::StructType>(op.getInput().getType());
    auto fieldIndex = op.getFieldIndex();
    auto elements = structType.getElements();

    int64_t totalBitWidth = hw::getBitWidth(structType);
    if (totalBitWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown struct width");

    int64_t consumedBits = 0;
    for (size_t i = 0; i < fieldIndex; ++i) {
      int64_t fieldWidth = hw::getBitWidth(elements[i].type);
      if (fieldWidth < 0)
        return rewriter.notifyMatchFailure(op.getLoc(), "unknown field width");
      consumedBits += fieldWidth;
    }
    int64_t fieldWidth = hw::getBitWidth(elements[fieldIndex].type);
    if (fieldWidth < 0)
      return rewriter.notifyMatchFailure(op.getLoc(), "unknown field width");

    int64_t bitOffset = totalBitWidth - consumedBits - fieldWidth;
    auto loc = op.getLoc();
    // comb.concat's first operand is the MSBs.
    SmallVector<Value> parts;
    if (bitOffset + fieldWidth < totalBitWidth)
      parts.push_back(comb::ExtractOp::create(
          rewriter, loc, adaptor.getInput(), bitOffset + fieldWidth,
          totalBitWidth - bitOffset - fieldWidth));
    parts.push_back(adaptor.getNewValue());
    if (bitOffset > 0)
      parts.push_back(
          comb::ExtractOp::create(rewriter, loc, adaptor.getInput(), 0,
                                  bitOffset));
    if (parts.size() == 1)
      rewriter.replaceOp(op, parts[0]);
    else
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, parts);
    return success();
  }
};

struct MuxOpConversion : OpConversionPattern<comb::MuxOp> {
  using OpConversionPattern<comb::MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Re-create Mux with legalized types.
    rewriter.replaceOpWithNewOp<comb::MuxOp>(
        op, adaptor.getCond(), adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// A type converter is needed to perform the in-flight materialization of
/// aggregate types to integer types.
class AggregateTypeConverter : public TypeConverter {
public:
  AggregateTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](hw::ArrayType t) -> Type {
      return IntegerType::get(t.getContext(), hw::getBitWidth(t));
    });
    addConversion([](hw::StructType t) -> Type {
      return IntegerType::get(t.getContext(), hw::getBitWidth(t));
    });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });

    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return Value();

      return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
          ->getResult(0);
    });
  }
};
} // namespace

static void populateHWAggregateToCombOpConversionPatterns(
    RewritePatternSet &patterns, AggregateTypeConverter &typeConverter,
    unsigned decodedArrayGetMin) {
  patterns.add<HWArrayGetOpConversion>(typeConverter, patterns.getContext(),
                                       decodedArrayGetMin);
  patterns.add<HWArrayCreateLikeOpConversion<hw::ArrayCreateOp>,
               HWArrayCreateLikeOpConversion<hw::ArrayConcatOp>,
               HWAggregateConstantOpConversion, HWArraySliceOpConversion,
               HWArrayInjectOpConversion, HWStructCreateOpConversion,
               HWStructExtractOpConversion, HWStructInjectOpConversion,
               MuxOpConversion>(
      typeConverter, patterns.getContext());
}

namespace {
struct HWAggregateToCombPass
    : public hw::impl::HWAggregateToCombBase<HWAggregateToCombPass> {
  void runOnOperation() override;
  using HWAggregateToCombBase<HWAggregateToCombPass>::HWAggregateToCombBase;
};
} // namespace

/// Ops inside an LLHD region (an unlowered `llhd.process` or
/// `llhd.combinational`) are off limits: the conversion's materializations
/// land at module scope, and an `llhd.wait`'s operands must be defined
/// inside its region -- rewriting through the boundary produces
/// "'llhd.wait' op using value defined outside the region". Checked by
/// dialect namespace to avoid a dependency on LLHD.
static bool isInsideLLHDRegion(Operation *op) {
  for (auto *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getDialectNamespace() == "llhd")
      return true;
  return false;
}

void HWAggregateToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<hw::ArrayGetOp, hw::ArrayCreateOp,
                               hw::ArrayConcatOp, hw::AggregateConstantOp,
                               hw::ArrayInjectOp, hw::ArraySliceOp,
                               hw::StructCreateOp, hw::StructExtractOp,
                               hw::StructInjectOp>(
      [](Operation *op) { return isInsideLLHDRegion(op); });
  target.addDynamicallyLegalOp<comb::MuxOp>([](comb::MuxOp op) {
    return hw::type_isa<IntegerType>(op.getType()) || isInsideLLHDRegion(op);
  });
  target.addLegalDialect<hw::HWDialect, comb::CombDialect>();

  RewritePatternSet patterns(&getContext());
  AggregateTypeConverter typeConverter;
  populateHWAggregateToCombOpConversionPatterns(patterns, typeConverter,
                                               decodedArrayGetMin);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
