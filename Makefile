# Compilers
CXX = g++
NVCC = nvcc

# Compilers flags
CXX_FLAGS = -std=c++17 -pthread -O3 -Wall -Wextra
NVCC_FLAGS = -gencode=arch=compute_61,code=compute_61 -O3
CUDA_FLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Target names
CPU_TARGET = $(BIN_DIR)/speed_cpu
GPU_TARGET = $(BIN_DIR)/speed_gpu

# Objects
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/reader.o $(BUILD_DIR)/tq.o
CPU_OBJS = $(OBJS) $(BUILD_DIR)/model.o
GPU_OBJS = $(OBJS) $(BUILD_DIR)/model_gpu.o

.PHONY: all init clean


# Targets
all: init $(CPU_TARGET) $(GPU_TARGET)

# Init
init:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build CPU target
$(CPU_TARGET): $(CPU_OBJS)
	$(CXX) $(CXX_FLAGS) $(CPU_OBJS) -o $(CPU_TARGET)

# Build GPU target
$(GPU_TARGET): $(GPU_OBJS)
	$(CXX) $(CXX_FLAGS) $(GPU_OBJS) -o $(GPU_TARGET) $(CUDA_FLAGS)

# Compile .o files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Compile GPU model
$(BUILD_DIR)/model_gpu.o:
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/model_gpu.o -c $(SRC_DIR)/model.cu

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)