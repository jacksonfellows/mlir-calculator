# get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
# get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# set(LIBS
        # ${dialect_libs}
        # ${conversion_libs}
        # MLIROptLib
        # MLIRStandalone
        # )
add_llvm_executable(mathc mathc.cpp)

llvm_update_compile_flags(mathc)
target_link_libraries(mathc PRIVATE MLIRSupport)

mlir_check_all_link_libraries(mathc)
