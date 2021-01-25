#include "Calculator/Parser.h"
#include "mlir/IR/Diagnostics.h"

// the left binding power of a token
int Parser::lbp(Token t) {
  switch (t) {
  case tok_num:
    return 0; // doesn't make sense
  case tok_add:
  case tok_sub:
    return 10;
  case tok_mul:
  case tok_div:
    return 20;
  case tok_pow:
    return 30;
  case tok_lparen:
    return 0;
  case tok_rparen:
    return 0;
  case tok_eof:
    return 0;
  }
}

void Parser::match(Token t) {
  if (t != token) {
    mlir::emitError(getLoc(), "Expected token " + lexer.tokToString(t) +
                                  ", instead encountered token " +
                                  lexer.tokToString(token));
    exit(1);
  }
  token = lexer.nextToken();
}

mlir::Location Parser::getLoc() {
  return generator.builder.getFileLineColLoc(
      generator.builder.getIdentifier("-"), lexer.getCurrentLine(),
      lexer.getLastCol());
}

mlir::Value Parser::nud(Token t, mlir::Location loc) {
  mlir::Value x;
  switch (t) {
  case tok_num:
    return generator.builder.create<mlir::ConstantFloatOp>(
        loc, llvm::APFloat(lexer.getCurrentNum()),
        generator.builder.getF64Type());
  case tok_sub:
    return generator.builder.create<mlir::NegFOp>(loc, expr(100));
  case tok_lparen:
    x = expr();
    match(tok_rparen);
    return x;
  default:
    mlir::emitError(loc, "No prefix handler for token " + lexer.tokToString(t));
    exit(1);
  }
}

mlir::Value Parser::led(Token t, mlir::Value left, mlir::Location loc) {
  switch (t) {
  case tok_add:
    return generator.builder.create<mlir::AddFOp>(loc, left, expr(10));
  case tok_sub:
    return generator.builder.create<mlir::SubFOp>(loc, left, expr(10));
  case tok_mul:
    return generator.builder.create<mlir::MulFOp>(loc, left, expr(20));
  case tok_div:
    return generator.builder.create<mlir::DivFOp>(loc, left, expr(20));
  case tok_pow:
    return generator.builder.create<mlir::PowFOp>(loc, left, expr(29));
  default:
    mlir::emitError(loc, "No infix handler for token " + lexer.tokToString(t));
    exit(1);
  }
}

mlir::Value Parser::expr(int rbp) {
  Token t = token;
  mlir::Location loc = getLoc();
  token = lexer.nextToken();
  mlir::Value left = nud(t, loc);
  while (rbp < lbp(token)) {
    t = token;
    mlir::Location loc = getLoc();
    token = lexer.nextToken();

    left = led(t, left, loc);
  }
  return left;
}

mlir::ModuleOp Parser::parse() {
  mlir::FuncOp mainFunc = mlir::FuncOp::create(
      generator.builder.getUnknownLoc(), "main",
      generator.builder.getFunctionType(llvm::None, llvm::None));
  mlir::Block &entryBlock = *mainFunc.addEntryBlock();
  generator.builder.setInsertionPointToStart(&entryBlock);

  token = lexer.nextToken();
  mlir::Value result = expr();
  match(tok_eof); // consume all available input

  generator.builder.create<mlir::calculator::PrintOp>(
      generator.builder.getUnknownLoc(), result);

  generator.builder.create<mlir::ReturnOp>(generator.builder.getUnknownLoc());

  generator.theModule.push_back(mainFunc);

  return generator.theModule;
}
