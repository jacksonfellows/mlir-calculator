#include "llvm/Support/raw_ostream.h"

#include "Calculator/Lexer.h"

unsigned int currentLine = 1;
unsigned int Lexer::getCurrentLine() { return currentLine; }

unsigned int lastCol = 0;
unsigned int currentCol = 0;
unsigned int Lexer::getLastCol() { return lastCol + 1; }

char currentNumStr[64];

llvm::StringRef Lexer::tokToString(Token t) {
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
    return currentNumStr;
  case tok_lparen:
    return "(";
  case tok_rparen:
    return ")";
  case tok_eof:
    return "EOF";
  }
}

double currentNum;
double Lexer::getCurrentNum() { return currentNum; }

Token Lexer::nextToken() {
  int charsRead;
  lastCol = currentCol;
  if (currentCol >= lineLen)
    return tok_eof;
  switch (line[currentCol++]) {
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
    currentCol--;
    sscanf(&line[currentCol], "%lf%n", &currentNum, &charsRead);
    memcpy(currentNumStr, line + currentCol, charsRead);
    currentNumStr[charsRead] = '\0';
    currentCol += charsRead;
    return tok_num;
  case '(':
    return tok_lparen;
  case ')':
    return tok_rparen;
  default:
    llvm::errs() << "Encountered unexpected character '" << line[currentCol - 1]
                 << "' while lexing.\n";
    exit(1);
  }
}
