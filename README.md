# MNIST Classification using cuDNN and cuBLAS 

## Overview

This project implements several neural network layers including:

- Convolutional layers: Implemented using cuDNN
- Activation layers: Implemented using cuDNN
- Pooling layers: Implemented using cuDNN
- Fully-connetced layers: Implemented using cuBLAS
- Dropout layers: Implemented using cuDNN

All these layers are implemented in `./advanced/src/layers/` directory. These layers are used to construct a custom network defined in `./advanced/src/models/cnn.cu` to classify MNIST database. The weights and biases are updated using Adam optimizer implemented in `./advanced/src/models/adam.cu`. You can add other optimizers in this file and use them to update weights as well.

The hyperparamters of this network such as filter height, number of filters, strides, padding, and etc. can be specified in the make file `Makefile` using the corresponding arguments (Refer to the make file for arguments description and how to use them.).

An augmentor class is also implemented using NVIDIA Performance Primitives (NPP) library functions to augment images if necessary. Currently, resize, normalization, and conversion from HWC to CHW format is implemented.

The code is commented and many classes can be used as a template to implement other features, such as different pooling, activation functions, adding/removing layers, and etc.

The `./simple/` folder also contains the forward pass of convolution, pooling, and activation layers using raw CUDA functions. 

After running experiments, the log files and weights can be found in `./output/` folder.

## Code Organization

- **`./bin/`**: Contains binary/executable files that are built automatically or manually.
- **`./data/`**: Holds the MNIST database.
- **`./advanced/lib/`**: Includes libraries header files.
- **`./advanced/src/`**: Contains the source files for the project.
- **`Makefile`**: Defines build and run commands for the project.

## Key Concepts

- Performance Strategies
- Image Processing
- Convolutional Neural Networks
- NPP, cuDNN and cuBLAS libraries
- CUDA 

## Supported OSes

- Linux
- Windows

## Dependencies

- [FreeImage](https://freeimage.sourceforge.io/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Ensure correct paths to include and library files for these dependencies are specified in the Makefile.

## Prerequisites

1. Download and install the [CUDA Toolkit 12.5](https://developer.nvidia.com/cuda-downloads) for your platform.
2. Install dependencies as listed in the [Dependencies](#dependencies) section.

## Building the Program

To build the project, use the following command:

```bash
make all
```

## Running the Program
After building the project, you can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary.

If you wish to run the binary directly with custom flags, you can use:

```bash
./bin/cnn_mnist.exe -d ./data/ 
```

You can change flags inside `Makefile` as well.

## Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
make clean
```

This will remove all files in the `./bin/` and `./output/` directory.
