#ifndef LOWER_H
#define LOWER_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Calculator/CalculatorOps.h"

namespace {
struct MathToLLVMLoweringPass
    : public mlir::PassWrapper<MathToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

#endif // LOWER_H
