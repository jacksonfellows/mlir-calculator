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
  Lexer lexer;

  Token token;
  mlir::Value expr(int rbp = 0);

  int lbp(Token);
  void match(Token);
  mlir::Location getLoc();
  mlir::Value nud(Token, mlir::Location);
  mlir::Value led(Token, mlir::Value, mlir::Location);

public:
  Parser(mlir::MLIRContext &context) : generator(context), lexer() {}

  mlir::ModuleOp parse();
};

#endif // PARSER_H
