//===- CalculatorOps.td - Calculator dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALCULATOR_OPS
#define CALCULATOR_OPS

include "CalculatorDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Calculator_PrintOp : Calculator_Op<"print"> {
    let arguments = (ins F64:$input);
}

#endif // CALCULATOR_OPS
