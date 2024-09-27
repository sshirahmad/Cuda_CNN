#ifndef CNNLayer_H
#define CNNLayer_H

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

class CNNLayer {
public:
    // Constructor
    CNNLayer(cudnnHandle_t cudnn, int inputHeight, int inputWidth,
            int filterHeight, int filterWidth,
            int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth,
            int outputChannels, int inputChannels,
            int batchSize, float learningrate);

    // Destructor
    ~CNNLayer();

    // Forward pass
    float* ForwardPass(const float* deviceInput);

    float* BackwardPass(const float* deviceOutputGrad);

    // Setters and Getters for parameters
    std::tuple<int, int, float*> GetOutput(int index);

private:
    int inputHeight, inputWidth;
    int filterHeight, filterWidth;
    int strideHeight, strideWidth;
    int paddingHeight, paddingWidth;
    int outputChannels, inputChannels;
    int poolWidth, poolHeight;
    int convHeight, convWidth;
    int batchSize;
    float learningrate;
    cudnnHandle_t cudnn;

    // Create tensor descriptors for input, output, and filters
    cudnnTensorDescriptor_t inputDesc, outputconvDesc, outputpoolDesc, filterTensorDesc;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t activationDesc;

    const float* deviceInput;
    float* deviceConv;
    float* deviceAct;
    float* deviceOutput;
    float* deviceFilter;

    float* deviceActGrad;
    float* deviceConvGrad;
    float* deviceInputGrad;
    float* deviceFilterGrad;
    const float* deviceOutputGrad;

    void CreateandSetDescs();
    void FreeMemory();
    void SetFilters();
    void LaunchConvolutionKernel();
    void LaunchActivationKernel();
    void LaunchMaxPoolingKernel();
    void LaunchBackwardMaxPoolingKernel();
    void LaunchBackwardActivationKernel();
    void LaunchBackwardConvolutionKernel();
    void UpdateWeights();

};

#endif // CNNLayer_H
