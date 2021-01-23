#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

#include "Calculator/LLVMHelpers.h"
#include "Calculator/Lower.h"
#include "Calculator/Parser.h"

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "calculator compiler\n");

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::calculator::CalculatorDialect>();

  Parser parser = Parser(context);
  mlir::ModuleOp theModule = parser.parse();

  if (failed(mlir::verify(theModule))) {
    theModule.emitError("module failed to verify");
    return 1;
  }

  // dump MLIR
  // TODO can I dump with loc info?
  theModule.dump();

  // perform canonicalization
  mlir::PassManager pm1(&context);

  mlir::OpPassManager &optPM = pm1.nest<mlir::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm1.run(theModule)))
    return 1;

  theModule.dump();

  // convert to LLVM IR
  mlir::PassManager pm2(&context);
  pm2.addPass(createLowerToLLVMPass());

  if (mlir::failed(pm2.run(theModule)))
    return 1;

  if (failed(mlir::verify(theModule))) {
    theModule.emitError("(lowered) module failed to verify");
    return 1;
  }

  // dump (lowered) MLIR
  theModule.dump();

  // dump LLVM IR
  dumpLLVMIR(theModule);

  // run JIT
  runJit(theModule);

  return 0;
}
