# XOR Example Makefile
# This is a convenience Makefile for building just the XOR examples

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall

# Directories
ROOT_DIR = ..
OUTPUT_DIR = ./output
OBJ_DIR = $(OUTPUT_DIR)/obj

# Target executables
TARGETS = $(OUTPUT_DIR)/xor_example $(OUTPUT_DIR)/advanced_example

# Object files
NEURAL_OBJ = $(OBJ_DIR)/neural_network.o

# Default target
all: dirs $(TARGETS)

# Create output directories
dirs:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(OBJ_DIR)

# Compile neural network framework
$(NEURAL_OBJ): $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.cpp $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.hpp | dirs
	$(CXX) $(CXXFLAGS) -c $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.cpp -o $(NEURAL_OBJ)

# Build XOR example
$(OUTPUT_DIR)/xor_example: xor_example.cpp $(NEURAL_OBJ) | dirs
	$(CXX) $(CXXFLAGS) xor_example.cpp $(NEURAL_OBJ) -o $(OUTPUT_DIR)/xor_example

# Build advanced example
$(OUTPUT_DIR)/advanced_example: advanced_example.cpp $(NEURAL_OBJ) | dirs
	$(CXX) $(CXXFLAGS) advanced_example.cpp $(NEURAL_OBJ) -o $(OUTPUT_DIR)/advanced_example

# Clean build artifacts
clean:
	rm -rf $(OUTPUT_DIR)

# Run the XOR example
run: $(OUTPUT_DIR)/xor_example
	@clear
	@./$(OUTPUT_DIR)/xor_example

# Run the advanced example
run-advanced: $(OUTPUT_DIR)/advanced_example
	@clear
	@./$(OUTPUT_DIR)/advanced_example

# Build and run XOR
build-run: all
	@clear
	@./$(OUTPUT_DIR)/xor_example

# Build and run advanced
build-run-advanced: all
	@clear
	@./$(OUTPUT_DIR)/advanced_example

# Phony targets
.PHONY: all clean run run-advanced build-run build-run-advanced dirs