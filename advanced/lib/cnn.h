#ifndef CNN_H
#define CNN_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cudnn.h>
#include <convolution.h>
#include <pooling.h>
#include <activation.h>
#include <fclayer.h>
#include <kernel.h>
#include <loss.h>

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

class CNN {
public:
    // Constructor
    CNN(cudnnHandle_t cudnn, cublasHandle_t cublas,
            int inputHeight, int inputWidth,
            int filterHeight, int filterWidth,
            int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth,
            int numFilter, int inputChannels,
            int hiddenDim, int numClass,
            int batchSize, float learningrate);

    // Destructor
    ~CNN();

    // Forward pass
    float* ForwardPass(const float* hostInput, const int* hostLabels);

    float* ComputeLoss();

    void BackwardPass();

    // Setters and Getters for parameters
    std::tuple<int, int, float*> GetOutput(int index);

private:
    int inputHeight, inputWidth;
    int filterHeight, filterWidth;
    int strideHeight, strideWidth;
    int paddingHeight, paddingWidth;
    int numFilter, inputChannels;
    int hiddenDim, numClass;
    int poolWidth, poolHeight;
    int convHeight, convWidth;
    int batchSize;
    int outputHeight, outputWidth, outputChannels;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    float learningrate;
    int flattenedSize;

    // CNN Layers
    ConvolutionLayer* C1 = nullptr;
    ActivationLayer* A1 = nullptr;
    PoolingLayer* P1 = nullptr;
    ConvolutionLayer* C2 = nullptr;
    ActivationLayer* A2 = nullptr;
    PoolingLayer* P2 = nullptr;
    ConvolutionLayer* C3 = nullptr;
    ActivationLayer* A3 = nullptr;
    PoolingLayer* P3 = nullptr;
    FCLayer* F4 = nullptr;
    ActivationLayer* A4 = nullptr;
    FCLayer* F5 = nullptr;

    float* deviceInput;
    float* deviceLoss;
    int* deviceLabels;

    float* cnnOutput;
    float* flattenedOutput;
    float* fcLogits;
    float* deviceOutputGrad;

    void AllocateMemory();
    void FreeMemory();
    void UpdateFilters();
    void BuildModel();
    std::tuple<int, int> CalculateDim(int inHeight, int inWidth);

};

#endif // CNN_H
