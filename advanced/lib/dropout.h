#ifndef DropoutLayer_H
#define DropoutLayer_H

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

class DropoutLayer {
public:
    // Constructor
    DropoutLayer(cudnnHandle_t cudnn,
            int inputHeight, int inputWidth,
            int inputChannels, int batchSize,
            float dropoutProbability);

    // Destructor
    ~DropoutLayer();

    // Forward pass
    float* ForwardPass(const float* deviceInput, bool training);

    float* BackwardPass(const float* deviceOutputGrad);

private:
    int inputHeight, inputWidth;
    int batchSize, inputChannels;
    float dropoutProbability;
    cudnnHandle_t cudnn;

    // Create tensor descriptors for input, output, and filters
    cudnnTensorDescriptor_t inputDesc;
    cudnnDropoutDescriptor_t dropoutDesc;

    const float* deviceInput;
    float* deviceOutput;
    float* dropoutMask;
    float* states;
    float* deviceInputGrad;
    const float* deviceOutputGrad;

    void CreateandSetDescs();
    void FreeMemory();
    void LaunchDropoutKernel(bool training);
    void LaunchBackwardDropoutKernel();

};

#endif // DropoutLayer_H
