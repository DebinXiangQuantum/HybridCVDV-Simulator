# Makefile for GPU Performance Test

# Compiler settings
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60 -std=c++17
LIBS = -lcublas -lcusparse

# Directories
SRC_DIR = .
GPU_DIR = $(SRC_DIR)/gpu
BUILD_DIR = $(SRC_DIR)/build

# Source files
GPU_SOURCES = $(GPU_DIR)/operators.cu
TEST_SOURCES = $(SRC_DIR)/gpu_performance_test.cpp

# Object files
GPU_OBJECTS = $(BUILD_DIR)/operators.o
TEST_OBJECTS = $(BUILD_DIR)/gpu_performance_test.o

# Executable
EXECUTABLE = $(BUILD_DIR)/gpu_performance_test

# Targets
all: $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/operators.o: $(GPU_DIR)/operators.cu $(GPU_DIR)/operators.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/gpu_performance_test.o: $(TEST_SOURCES) $(GPU_DIR)/operators.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXECUTABLE): $(GPU_OBJECTS) $(TEST_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

run:
	$(EXECUTABLE)

.PHONY: all clean run
