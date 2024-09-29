#ifndef PoolingLayer_H
#define PoolingLayer_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cudnn.h>
#include <kernel.h>

// Error checking macro for cuDNN calls
#define CHECK_CUDNN(call) \
    {                       \
        cudnnStatus_t status = call;                                        \
        if ((call) != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error: " << status << " in "   \
            << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }


#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " \
                      << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                      \
    }

class PoolingLayer {
public:
    // Constructor
    PoolingLayer(cudnnHandle_t cudnn,
                int inputHeight, int inputWidth,
                int filterHeight, int filterWidth,
                int strideHeight, int strideWidth,
                int paddingHeight, int paddingWidth,
                int inputChannels, int batchSize);

    // Destructor
    ~PoolingLayer();

    // Forward pass
    float* ForwardPass(const float* deviceInput);

    float* BackwardPass(const float* deviceOutputGrad);

private:
    int inputHeight, inputWidth;
    int filterHeight, filterWidth;
    int strideHeight, strideWidth;
    int paddingHeight, paddingWidth;
    int inputChannels;
    int poolWidth, poolHeight;
    int batchSize;
    cudnnHandle_t cudnn;

    // Create tensor descriptors for input, output, and filters
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnPoolingDescriptor_t poolDesc;


    const float* deviceInput;
    float* deviceOutput;
    float* deviceInputGrad;
    const float* deviceOutputGrad;

    void CreateandSetDescs();
    void FreeMemory();
    void LaunchMaxPoolingKernel();
    void LaunchBackwardMaxPoolingKernel();

};

#endif // PoolingLayer_H
