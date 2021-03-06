#ifndef CALCULATOR_LEXER_H
#define CALCULATOR_LEXER_H

#include <stdio.h>

#include "llvm/ADT/StringRef.h"

enum Token {
  // operators
  tok_add,
  tok_sub,
  tok_mul,
  tok_div,
  tok_pow,

  // literals
  tok_num,

  // parens
  tok_lparen,
  tok_rparen,

  // other
  tok_eof
};

class Lexer {
  char *line = NULL;
  ssize_t lineLen;

public:
  Lexer() {
    size_t lineCap = 0;
    lineLen = getline(&line, &lineCap, stdin);
  }

  llvm::StringRef tokToString(Token);

  unsigned int getCurrentLine();
  unsigned int getLastCol();

  double getCurrentNum();

  Token nextToken();
};

#endif // CALCULATOR_LEXER_H
