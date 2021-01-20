#include "llvm/Support/raw_ostream.h"

#include "Calculator/Lexer.h"

unsigned int currentLine = 1;
unsigned int getCurrentLine() { return currentLine; }

unsigned int lastCol = 1;
unsigned int currentCol = 1;
unsigned int getLastCol() { return lastCol; }

llvm::StringRef tokToString(Token t) {
  switch (t) {
  case tok_add:
    return "+";
  case tok_sub:
    return "-";
  case tok_mul:
    return "*";
  case tok_div:
    return "/";
  case tok_pow:
    return "^";
  case tok_num:
    return "[number]";
  case tok_lparen:
    return "(";
  case tok_rparen:
    return ")";
  case tok_eof:
    return "EOF";
  }
}

double currentNum;
double getCurrentNum() { return currentNum; }

Token nextToken() {
  int charsRead;
  lastCol = currentCol;
  currentCol++;
  switch (int c = getchar()) {
  case ' ':
  case '\n':
    return nextToken();
  case '+':
    return tok_add;
  case '-':
    return tok_sub;
  case '*':
    return tok_mul;
  case '/':
    return tok_div;
  case '^':
    return tok_pow;
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    ungetc(c, stdin);
    scanf("%lf%n", &currentNum, &charsRead);
    currentCol += charsRead - 1;
    return tok_num;
  case '(':
    return tok_lparen;
  case ')':
    return tok_rparen;
  case EOF:
    return tok_eof;
  default:
    llvm::errs() << "Encountered unexpected character '" << (char)c
                 << "' while lexing.\n";
    exit(1);
  }
}
