#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

#include "Calculator/LLVMHelpers.h"
#include "Calculator/Lower.h"
#include "Calculator/Parser.h"

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions(); // can control e.g. how MLIR is printed
  llvm::cl::ParseCommandLineOptions(argc, argv, "calculator compiler\n");

  // create a context and load the dialects that we need (std + calculator)
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::calculator::CalculatorDialect>();

  // parse the math expression from stdin into a module
  Parser parser = Parser(context);
  mlir::ModuleOp theModule = parser.parse();

  if (failed(mlir::verify(theModule))) {
    theModule.emitError("module failed to verify");
    return 1;
  }

  // dump MLIR of module
  theModule.dump();

  // perform canonicalization pass
  mlir::PassManager pm1(&context);

  mlir::OpPassManager &optPM = pm1.nest<mlir::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm1.run(theModule)))
    return 1;

  theModule.dump();

  // convert std + calculator to llvm
  mlir::PassManager pm2(&context);
  pm2.addPass(createLowerToLLVMPass());

  if (mlir::failed(pm2.run(theModule)))
    return 1;

  if (failed(mlir::verify(theModule))) {
    theModule.emitError("(lowered to llvm) module failed to verify");
    return 1;
  }

  // dump (lowered) MLIR
  theModule.dump();

  // dump LLVM IR
  dumpLLVMIR(theModule);

  // run via LLVM's JIT compiler
  runJit(theModule);

  return 0;
}
