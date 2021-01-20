#ifndef GENERATOR_H
#define GENERATOR_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

class MLIRGenerator {
public:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

  MLIRGenerator(mlir::MLIRContext &context) : builder(&context) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
};

#endif // GENERATOR_H
