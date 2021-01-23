add_mlir_dialect_library(MLIRCalculator
        CalculatorDialect.cpp
        CalculatorOps.cpp
        Lexer.cpp
        Parser.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Calculator

        DEPENDS
        MLIRCalculatorOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
