//===- CalculatorDialect.td - Calculator dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALCULATOR_DIALECT
#define CALCULATOR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Calculator dialect definition.
//===----------------------------------------------------------------------===//

def Calculator_Dialect : Dialect {
    let name = "calculator";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::calculator";
}

//===----------------------------------------------------------------------===//
// Base calculator operation definition.
//===----------------------------------------------------------------------===//

class Calculator_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Calculator_Dialect, mnemonic, traits>;

#endif // CALCULATOR_DIALECT
