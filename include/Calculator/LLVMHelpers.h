#ifndef LLVMHELPERS_H
#define LLVMHELPERS_H

#include "mlir/IR/BuiltinOps.h"

int runJit(mlir::ModuleOp);

int dumpLLVMIR(mlir::ModuleOp);

#endif // LLVMHELPERS_H
