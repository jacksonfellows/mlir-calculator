# A simple calculator implemented with MLIR

This is an example of simple calculator implemented using [MLIR](https://mlir.llvm.org/).

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To setup CMake to build the project, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
```
To build `calculatorc` (the calculator compiler), run
```sh
cmake --build . --target calculatorc
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Usage

`calculatorc` parses and compiles a single expression from `stdin`.
It supports the standard operators `+`, `-`, `*`, `/`, and `^` (exponentiation), along with `(` and `)` for grouping.
The precedence and associativity of these operators should behave as expected.
Numbers are inputted as decimals, optionally with scientific notation (e.g. `3.5e4` for `35000`).
Internally, all numbers are represented as `double`s.
E.g. the following should compute and print `1022.75`:
```sh
echo '(2 - 3) * 5 / 4 + 4^5' | ./bin/calculatorc
```
