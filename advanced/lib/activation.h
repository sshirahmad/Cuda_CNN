#ifndef ActivationLayer_H
#define ActivationLayer_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cudnn.h>
#include <kernel.h>

// Error checking macro for cuDNN calls
#define CHECK_CUDNN(call) \
    if ((call) != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
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

class ActivationLayer {
public:
    // Constructor
    ActivationLayer(cudnnHandle_t cudnn,
            int inputHeight, int inputWidth,
            int inputChannels, int batchSize);

    // Destructor
    ~ActivationLayer();

    // Forward pass
    float* ForwardPass(const float* deviceInput);

    float* BackwardPass(const float* deviceOutputGrad);

private:
    int inputHeight, inputWidth;
    int batchSize, inputChannels;
    cudnnHandle_t cudnn;

    // Create tensor descriptors for input, output, and filters
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnActivationDescriptor_t activationDesc;

    const float* deviceInput;
    float* deviceOutput;

    float* deviceInputGrad;
    const float* deviceOutputGrad;

    void CreateandSetDescs();
    void FreeMemory();
    void LaunchActivationKernel();
    void LaunchBackwardActivationKernel();

};

#endif // ActivationLayer_H
