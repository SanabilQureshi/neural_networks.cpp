# Superstore Example Makefile
# This is a convenience Makefile for building just the Superstore example

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall

# Directories
ROOT_DIR = ..
OUTPUT_DIR = ./output
OBJ_DIR = $(OUTPUT_DIR)/obj

# Target executable
TARGET = $(OUTPUT_DIR)/superstore_classification

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

# Build Superstore classification
$(TARGET): superstore_classification.cpp $(NEURAL_OBJ) | dirs
	$(CXX) $(CXXFLAGS) superstore_classification.cpp $(NEURAL_OBJ) -o $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OUTPUT_DIR)

# Run the Superstore example
run: $(TARGET)
	@clear
	@./$(TARGET)

# Build and run
build-run: all
	@clear
	@./$(TARGET)

# Phony targets
.PHONY: all clean run build-run dirs