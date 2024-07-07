# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++17 -O3 -pthread -Wall -Wextra -g
# Executable name
TARGET = speed_gpu
# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS)) $(BUILD_DIR)/model.o

# Targets
all: $(BIN_DIR)/$(TARGET)

$(BUILD_DIR)/model.o:
	nvcc -c $(SRC_DIR)/model.cu

# Linking rule
$(BIN_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compilation rule
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean