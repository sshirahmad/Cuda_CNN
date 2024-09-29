# Define directories
SRC_DIR = ./advanced/src
BIN_DIR = ./bin
OUTPUT_DIR = ./output
DATA_DIR = ./data
LIB_DIR = ./advanced/lib

# Define the compiler and flags
NVCC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe"

# without ccbin to x64 the compiler gives Access error (MUST FLGAG)
# -I flag is the path to header files
# -L flag is the path to lib files
CXXFLAGS = -std=c++17 \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include" \
           -I"C:/opencv/build/include" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/Common" \
		   -I"C:/Program Files/NVIDIA/CUDNN/v9.4/include/12.6" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/Common/UtilNPP" \
           -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/FreeImage/Dist/x64" \
		   -I"$(LIB_DIR)" \
           -ccbin "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64"

LDFLAGS = -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64" \
          -lcudart -lcublas -lnppc -lnppial -lnppif -lnppig -lnppist -lnppisu \
		  -L"C:/Program Files/NVIDIA/CUDNN/v9.4/lib/12.6/x64" \
          -lcudnn \
          -L"C:/opencv/build/x64/vc16/lib" \
          -lopencv_world4100 -lopencv_world4100d \
          -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/FreeImage/Dist/x64" \
          -lFreeImage



# Define source files and target executable
SRC = $(SRC_DIR)/cnn_mnist.cu $(SRC_DIR)/layers/convolution.cu $(SRC_DIR)/layers/dropout.cu $(SRC_DIR)/layers/pooling.cu $(SRC_DIR)/layers/activation.cu $(SRC_DIR)/layers/fclayer.cu $(SRC_DIR)/layers/kernel.cu $(SRC_DIR)/data_utils/augmentations.cu $(SRC_DIR)/models/cnn.cu $(SRC_DIR)/models/loss.cu $(SRC_DIR)/models/adam.cu 

TARGET = $(BIN_DIR)/cnn_mnist.exe

# Define the default rule
all: $(TARGET)

# Rule for building the target executable
$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Rule for running the application
run: $(TARGET)
	$(TARGET) -d "../data/" -ds "./output/weights/" -dl "./output/weights/weights_best.bin" -w 28 -h 28 -cfh 3 -cfw 3 -csh 1 -csw 1 -cph 1 -cpw 1 -pfh 2 -pfw 2 -psh 1 -psw 1 -pph 0 -ppw 0 -nf 16 -hd 32 -nc 10 -bs 128 -lr 0.0001 -e 300 -wd 0.0 -dp 0.5 > ./output/output_log.txt 2>&1


# Clean up
clean:
	rm -rf $(BIN_DIR)/* $(OUTPUT_DIR)/* 

# Installation rule (not much to install, but here for completeness)
install:
	@echo "No installation required."

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make install- Install the project (if applicable)."
	@echo "  make help   - Display this help message."
