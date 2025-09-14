# MNIST Example Makefile
# This is a convenience Makefile for building just the MNIST example

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall

# Directories
ROOT_DIR = ..
OUTPUT_DIR = ./output
OBJ_DIR = $(OUTPUT_DIR)/obj
FRAMEWORK_DIR = $(ROOT_DIR)/Neural Network Framework

# Target executable
TARGET = $(OUTPUT_DIR)/mnist_example

# Object files
NEURAL_OBJ = $(OBJ_DIR)/neural_network.o

# Default target
all: dirs $(TARGET)

# Create output directories
dirs:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(OBJ_DIR)

# Compile neural network framework
$(NEURAL_OBJ): $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.cpp $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.hpp | dirs
	$(CXX) $(CXXFLAGS) -c $(ROOT_DIR)/Neural\ Network\ Framework/neural_network.cpp -o $(NEURAL_OBJ)

# Build MNIST example
$(TARGET): mnist_example.cpp $(NEURAL_OBJ) | dirs
	$(CXX) $(CXXFLAGS) mnist_example.cpp $(NEURAL_OBJ) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(NEURAL_OBJ)

# Run the MNIST example
run: $(TARGET)
	@clear
	@./$(TARGET)

# Build and run
build-run: all
	@clear
	@./$(TARGET)

# Phony targets
.PHONY: all clean run build-run dirs
