#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "Calculator/CalculatorDialect.h"
#include "Calculator/CalculatorOps.h"
#include "Calculator/Generator.h"
#include "Calculator/Lexer.h"

// parser/MLIR generator

mlir::MLIRContext context;

MLIRGenerator getGenerator() {
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::calculator::CalculatorDialect>();
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
    llvm::errs() << "Expected token " << tokToString(t)
                 << ", instead encountered token " << tokToString(token)
                 << "\n";
    exit(1);
  }
  token = nextToken();
}

mlir::Location getLoc() {
  return generator.builder.getFileLineColLoc(
      generator.builder.getIdentifier("-"), getCurrentLine(), getLastCol());
}

// prefix handler
mlir::Value nud(Token t, mlir::Location loc) {
  mlir::Value x;
  switch (t) {
  case tok_num:
    return generator.builder.create<mlir::ConstantFloatOp>(
        loc, llvm::APFloat(getCurrentNum()), generator.builder.getF64Type());
  case tok_sub:
    return generator.builder.create<mlir::NegFOp>(loc, expr(100));
  case tok_lparen:
    x = expr();
    match(tok_rparen);
    return x;
  default:
    llvm::errs() << "No prefix handler for token " << tokToString(t) << "\n";
    exit(1);
  }
}

// infix handler
mlir::Value led(Token t, mlir::Value left, mlir::Location loc) {
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
    llvm::errs() << "No infix handler for token " << tokToString(t) << ".\n";
    exit(1);
  }
}

mlir::Value expr(int rbp) {
  Token t = token;
  mlir::Location loc = getLoc();
  token = nextToken();
  mlir::Value left = nud(t, loc);
  while (rbp < lbp(token)) {
    t = token;
    mlir::Location loc = getLoc();
    token = nextToken();

    left = led(t, left, loc);
  }
  return left;
}

void parse() {
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

class PrintOpLowering : public mlir::ConversionPattern {
public:
  explicit PrintOpLowering(mlir::MLIRContext *context)
      : ConversionPattern(mlir::calculator::PrintOp::getOperationName(), 1,
                          context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", "%lf\n", parentModule);

    // Generate a call to printf for the current element of the loop.
    auto printOp = llvm::cast<mlir::calculator::PrintOp>(op);
    rewriter.create<mlir::CallOp>(
        loc, printfRef, rewriter.getIntegerType(32),
        llvm::ArrayRef<mlir::Value>({formatSpecifierCst, printOp.input()}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
      return mlir::SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmI8PtrTy =
        mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            llvmFnType);
    return mlir::SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             llvm::StringRef name,
                                             llvm::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(builder.getContext(), 8)),
        globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

namespace {
struct MathToLLVMLoweringPass
    : public mlir::PassWrapper<MathToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

void MathToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::OwningRewritePatternList patterns;
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  patterns.insert<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<MathToLLVMLoweringPass>();
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "calculator compiler\n");

  parse();
  if (failed(mlir::verify(generator.theModule))) {
    generator.theModule.emitError("module failed to verify");
    return 1;
  }

  // dump MLIR
  // TODO can I dump with loc info?
  generator.theModule.dump();

  // perform canonicalization
  mlir::PassManager pm1(&context);

  mlir::OpPassManager &optPM = pm1.nest<mlir::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm1.run(generator.theModule)))
    return 1;

  generator.theModule.dump();

  // convert to LLVM IR
  mlir::PassManager pm2(&context);
  pm2.addPass(createLowerToLLVMPass());

  if (mlir::failed(pm2.run(generator.theModule)))
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
