set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

add_llvm_executable(calculatorc calculatorc.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

llvm_update_compile_flags(calculatorc)
target_link_libraries(calculatorc
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRSupport
  MLIROptLib
  MLIRExecutionEngine
  MLIRTargetLLVMIR
  MLIRPass
  )

mlir_check_all_link_libraries(calculatorc)
