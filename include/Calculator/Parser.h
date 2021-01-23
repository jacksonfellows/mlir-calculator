#ifndef PARSER_H
#define PARSER_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "Calculator/CalculatorDialect.h"
#include "Calculator/CalculatorOps.h"
#include "Calculator/Generator.h"
#include "Calculator/Lexer.h"

class Parser {
  MLIRGenerator generator;

  Token token;
  mlir::Value expr(int rbp = 0);

  int lbp(Token);
  void match(Token);
  mlir::Location getLoc();
  mlir::Value nud(Token, mlir::Location);
  mlir::Value led(Token, mlir::Value, mlir::Location);

public:
  Parser(mlir::MLIRContext &context) : generator(context) {}

  mlir::ModuleOp parse() {
    mlir::FuncOp mainFunc = mlir::FuncOp::create(
        generator.builder.getUnknownLoc(), "main",
        generator.builder.getFunctionType(llvm::None, llvm::None));
    mlir::Block &entryBlock = *mainFunc.addEntryBlock();
    generator.builder.setInsertionPointToStart(&entryBlock);

    token = nextToken();
    mlir::Value result = expr();
    match(tok_eof); // consume all available input

    generator.builder.create<mlir::calculator::PrintOp>(
        generator.builder.getUnknownLoc(), result);

    generator.builder.create<mlir::ReturnOp>(generator.builder.getUnknownLoc());

    generator.theModule.push_back(mainFunc);

    return generator.theModule;
  }
};

#endif // PARSER_H
