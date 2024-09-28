#ifndef ConvolutionLayer_H
#define ConvolutionLayer_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cudnn.h>
#include <kernel.h>
#include <cublas_v2.h>
#include <vector>

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

#define CHECK_CUBLAS(call)                                                    \
    {                                                                          \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "CUBLAS error: " << status << " in " << __FILE__    \
                      << " at line " << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                      \
    }

class ConvolutionLayer {
public:
    // Constructor
    ConvolutionLayer(cudnnHandle_t cudnn, cublasHandle_t cublas,
                    int inputHeight, int inputWidth,
                    int filterHeight, int filterWidth,
                    int strideHeight, int strideWidth,
                    int paddingHeight, int paddingWidth,
                    int outputChannels, int inputChannels,
                    int batchSize, float learningrate);

    // Destructor
    ~ConvolutionLayer();

    // Forward pass
    float* ForwardPass(const float* deviceInput);

    float* BackwardPass(const float* deviceOutputGrad);

    void SaveWeights(FILE* file);

    void LoadWeights(FILE* file);

private:
    int inputHeight, inputWidth;
    int filterHeight, filterWidth;
    int strideHeight, strideWidth;
    int paddingHeight, paddingWidth;
    int outputChannels, inputChannels;
    int convHeight, convWidth;
    int batchSize;
    float learningrate;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    // Create tensor descriptors for input, output, and filters
    cudnnTensorDescriptor_t inputDesc, outputDesc, filterTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    const float* deviceInput;
    float* deviceOutput;
    float* deviceFilter;
    float* deviceInputGrad;
    float* deviceFilterGrad;
    const float* deviceOutputGrad;

    void CreateandSetDescs();
    void FreeMemory();
    void SetFilters();
    void LaunchConvolutionKernel();
    void LaunchBackwardConvolutionKernel();
    void UpdateWeights();

};

#endif // ConvolutionLayer_H
