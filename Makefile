# Makefile for Neural Network Framework
# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall

# Directories
OUTPUT_DIR = output
OBJ_DIR = $(OUTPUT_DIR)/obj

# Target executables
TARGETS = $(OUTPUT_DIR)/xor_example $(OUTPUT_DIR)/advanced_example $(OUTPUT_DIR)/mnist_example $(OUTPUT_DIR)/superstore_classification

# Object files
OBJS = $(OBJ_DIR)/neural_network.o

# Code formatter
FORMATTER = clang-format
FORMAT_STYLE = -style="{BasedOnStyle: Microsoft}"

# Default target: format and build everything
all: format dirs $(TARGETS)

# Create output directories
dirs:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(OBJ_DIR)

# Format all source files
format:
	@echo "Formatting source files..."
	@$(FORMATTER) $(FORMAT_STYLE) -i *.cpp *.hpp 2>/dev/null || true

# Compile neural_network.cpp into object file
$(OBJ_DIR)/neural_network.o: Neural\ Network\ Framework/neural_network.cpp Neural\ Network\ Framework/neural_network.hpp | dirs
	$(CXX) $(CXXFLAGS) -c Neural\ Network\ Framework/neural_network.cpp -o $(OBJ_DIR)/neural_network.o

# Build XOR example executable
$(OUTPUT_DIR)/xor_example: XOR/xor_example.cpp $(OBJ_DIR)/neural_network.o | dirs
	$(CXX) $(CXXFLAGS) XOR/xor_example.cpp $(OBJ_DIR)/neural_network.o -o $(OUTPUT_DIR)/xor_example

# Build advanced features example executable
$(OUTPUT_DIR)/advanced_example: XOR/advanced_example.cpp $(OBJ_DIR)/neural_network.o | dirs
	$(CXX) $(CXXFLAGS) XOR/advanced_example.cpp $(OBJ_DIR)/neural_network.o -o $(OUTPUT_DIR)/advanced_example

# Build MNIST example executable
$(OUTPUT_DIR)/mnist_example: MNIST/mnist_example.cpp $(OBJ_DIR)/neural_network.o | dirs
	$(CXX) $(CXXFLAGS) MNIST/mnist_example.cpp $(OBJ_DIR)/neural_network.o -o $(OUTPUT_DIR)/mnist_example

# Build Superstore classification executable
$(OUTPUT_DIR)/superstore_classification: Superstore/superstore_classification.cpp $(OBJ_DIR)/neural_network.o | dirs
	$(CXX) $(CXXFLAGS) Superstore/superstore_classification.cpp $(OBJ_DIR)/neural_network.o -o $(OUTPUT_DIR)/superstore_classification

# Clean all build artifacts
clean:
	rm -rf $(OUTPUT_DIR)

# Run the XOR example
run: $(OUTPUT_DIR)/xor_example
	@clear
	@cd XOR && ../$(OUTPUT_DIR)/xor_example

# Run the advanced example
run-advanced: $(OUTPUT_DIR)/advanced_example
	@clear
	@cd XOR && ../$(OUTPUT_DIR)/advanced_example

# Run the MNIST example
run-mnist: $(OUTPUT_DIR)/mnist_example
	@clear
	@cd MNIST && ../$(OUTPUT_DIR)/mnist_example

# Run the Superstore example
run-superstore: $(OUTPUT_DIR)/superstore_classification
	@clear
	@cd Superstore && ../$(OUTPUT_DIR)/superstore_classification

# Format and build, then run XOR (mimics your VSCode command)
build-run: format dirs $(OUTPUT_DIR)/xor_example
	@clear
	@cd XOR && ../$(OUTPUT_DIR)/xor_example

# Format and build, then run MNIST
build-run-mnist: format dirs $(OUTPUT_DIR)/mnist_example
	@clear
	@cd MNIST && ../$(OUTPUT_DIR)/mnist_example

# Format and build, then run Superstore
build-run-superstore: format dirs $(OUTPUT_DIR)/superstore_classification
	@clear
	@cd Superstore && ../$(OUTPUT_DIR)/superstore_classification

# Run all cases (XOR, MNIST, Superstore)
run-all: $(TARGETS)
	@echo "===== Running XOR Example ====="
	@cd XOR && ../$(OUTPUT_DIR)/xor_example
	@echo ""
	@echo "===== Running Advanced XOR Example ====="
	@cd XOR && ../$(OUTPUT_DIR)/advanced_example
	@echo ""
	@echo "===== Running MNIST Example ====="
	@cd MNIST && ../$(OUTPUT_DIR)/mnist_example
	@echo ""
	@echo "===== Running Superstore Classification ====="
	@cd Superstore && ../$(OUTPUT_DIR)/superstore_classification

# Phony targets (not actual files)
.PHONY: all clean run run-advanced run-mnist run-superstore format dirs build-run build-run-mnist build-run-superstore run-all