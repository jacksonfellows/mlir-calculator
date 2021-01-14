#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <stdio.h>

// lexer

unsigned int currentLine = 0;
unsigned int currentCol = 0;

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

double currentNum;

Token nextToken() {
  int charsRead;
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

// parser/MLIR generator

class MLIRGenerator {
public:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

  MLIRGenerator(mlir::MLIRContext &context) : builder(&context) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
};

mlir::MLIRContext context;

MLIRGenerator getGenerator() {
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>(); // need for printf
  return MLIRGenerator(context);
}

MLIRGenerator generator = getGenerator();

// the left binding power of a token
int lbp(Token t) {
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

Token token;

mlir::Value expr(int rbp = 0);

void match(Token t) {
  if (t != token) {
    llvm::errs() << "Expected other token.\n";
    exit(1);
  }
  token = nextToken();
}

mlir::Location getLoc() {
  return generator.builder.getFileLineColLoc(
      generator.builder.getIdentifier("-"), currentLine, currentCol);
}

// prefix handler
mlir::Value nud(Token t) {
  mlir::Value x;
  switch (t) {
  case tok_num:
    return generator.builder.create<mlir::ConstantFloatOp>(
        getLoc(), llvm::APFloat(currentNum), generator.builder.getF64Type());
  case tok_sub:
    return generator.builder.create<mlir::NegFOp>(getLoc(), expr(100));
  case tok_lparen:
    x = expr();
    match(tok_rparen);
    return x;
  default:
    llvm::errs() << "No prefix handler for token.\n";
    exit(1);
  }
}

mlir::Value led(Token t, mlir::Value left) {
  switch (t) {
  case tok_add:
    return generator.builder.create<mlir::AddFOp>(getLoc(), left, expr(10));
  case tok_sub:
    return generator.builder.create<mlir::SubFOp>(getLoc(), left, expr(10));
  case tok_mul:
    return generator.builder.create<mlir::MulFOp>(getLoc(), left, expr(20));
  case tok_div:
    return generator.builder.create<mlir::DivFOp>(getLoc(), left, expr(20));
  case tok_pow:
    return generator.builder.create<mlir::PowFOp>(getLoc(), left, expr(29));
  default:
    llvm::errs() << "No infix handler for token.\n";
    exit(1);
  }
}

mlir::Value expr(int rbp) {
  Token t = token;
  token = nextToken();
  mlir::Value left = nud(t);
  while (rbp < lbp(token)) {
    t = token;
    token = nextToken();
    left = led(t, left);
  }
  return left;
}

// taken from Toy tutorial

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
mlir::FlatSymbolRefAttr getOrInsertPrintf(mlir::OpBuilder &builder,
                                          mlir::ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
    return mlir::SymbolRefAttr::get("printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = mlir::LLVM::LLVMIntegerType::get(context, 32);
  auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(
      mlir::LLVM::LLVMIntegerType::get(context, 8));
  auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                      /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  mlir::PatternRewriter::InsertionGuard insertGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return mlir::SymbolRefAttr::get("printf", context);
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    llvm::StringRef name, llvm::StringRef value,
                                    mlir::ModuleOp module) {
  // Create the global at the entry of the module.
  mlir::LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = mlir::LLVM::LLVMArrayType::get(
        mlir::LLVM::LLVMIntegerType::get(builder.getContext(), 8),
        value.size());
    global = builder.create<mlir::LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
        builder.getStringAttr(value));
  }

  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
      loc, mlir::LLVM::LLVMIntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<mlir::LLVM::GEPOp>(
      loc,
      mlir::LLVM::LLVMPointerType::get(
          mlir::LLVM::LLVMIntegerType::get(builder.getContext(), 8)),
      globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
}

void parse() {
  mlir::FuncOp mainFunc = mlir::FuncOp::create(
      getLoc(), "main",
      generator.builder.getFunctionType(llvm::None, llvm::None));
  mlir::Block &entryBlock = *mainFunc.addEntryBlock();
  generator.builder.setInsertionPointToStart(&entryBlock);

  token = nextToken();
  mlir::Value result = expr();

  mlir::FlatSymbolRefAttr printfRef =
      getOrInsertPrintf(generator.builder, generator.theModule);
  mlir::Value fmt = getOrCreateGlobalString(getLoc(), generator.builder, "fmt",
                                            "%lf\n", generator.theModule);
  generator.builder.create<mlir::CallOp>(
      getLoc(), printfRef, generator.builder.getI32Type(),
      llvm::ArrayRef<mlir::Value>({fmt, result}));

  generator.builder.create<mlir::ReturnOp>(getLoc());

  generator.theModule.push_back(mainFunc);
}

bool enableOpt = 1;

// taken from Toy tutorial

int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {
  parse();
  if (failed(mlir::verify(generator.theModule))) {
    generator.theModule.emitError("module failed to verify");
    // return 1;
  }

  // dump MLIR
  // TODO can I dump with loc info?
  generator.theModule.dump();

  // lower to LLVM IR
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createLowerToLLVMPass());
  if (mlir::failed(pm.run(generator.theModule)))
    return 1;

  if (failed(mlir::verify(generator.theModule))) {
    generator.theModule.emitError("(lowered) module failed to verify");
    return 1;
  }

  // dump (lowered) MLIR
  generator.theModule.dump();

  // dump LLVM IR
  dumpLLVMIR(generator.theModule);

  // run JIT
  runJit(generator.theModule);

  return 0;
}
