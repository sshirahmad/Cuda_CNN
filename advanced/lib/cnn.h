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
#include <adam.h>
#include <dropout.h>

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

class CNN {
public:
    // Constructor
    CNN(cudnnHandle_t cudnn, cublasHandle_t cublas,
            int inputHeight, int inputWidth,
            int convfilterHeight, int convfilterWidth,
            int convstrideHeight, int convstrideWidth,
            int convpaddingHeight, int convpaddingWidth,
            int poolfilterHeight, int poolfilterWidth,
            int poolstrideHeight, int poolstrideWidth,
            int poolpaddingHeight, int poolpaddingWidth,
            int numFilter, int inputChannels,
            int hiddenDim, int numClass,
            int batchSize, float learningrate,
            float weight_decay, float dropoutProbability);

    // Destructor
    ~CNN();

    // Forward pass
    float* ForwardPass(const float* hostInput, const int* hostLabels, bool training);

    float* ComputeLoss();

    float* ComputeAccuracy();

    void BackwardPass();

    void SaveModelWeights(const std::string& filename); 

    void LoadModelWeights(const std::string& filename);

    // Setters and Getters for parameters
    std::tuple<int, int, float*> GetOutput(int index);

private:
    int inputHeight, inputWidth;
    int convfilterHeight, convfilterWidth;
    int convstrideHeight, convstrideWidth;
    int convpaddingHeight, convpaddingWidth;
    int poolfilterHeight, poolfilterWidth;
    int poolstrideHeight, poolstrideWidth;
    int poolpaddingHeight, poolpaddingWidth;
    int numFilter, inputChannels;
    int hiddenDim, numClass;
    int poolWidth, poolHeight;
    int convHeight, convWidth;
    int batchSize;
    int outputHeight, outputWidth, outputChannels;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    float learningrate;
    float weight_decay;
    float dropoutProbability;

    // CNN Layers
    ConvolutionLayer* C1 = nullptr;
    ActivationLayer* A1 = nullptr;
    PoolingLayer* P1 = nullptr;

    ConvolutionLayer* C2 = nullptr;
    ActivationLayer* A2 = nullptr;
    PoolingLayer* P2 = nullptr;

    ConvolutionLayer* C3 = nullptr;

    FCLayer* F3 = nullptr;
    ActivationLayer* A3 = nullptr;
    DropoutLayer* D3 = nullptr;

    FCLayer* F4 = nullptr;

    float* deviceInput;
    float* deviceLoss;
    int* deviceLabels;
    float* deviceAccuracy;

    float* cnnOutput;
    float* fcLogits;
    float* deviceOutputGrad;

    void AllocateMemory();
    void FreeMemory();
    void BuildModel();
    std::tuple<int, int> CalculateDim(int inHeight, int inWidth, std::string type);

};

#endif // CNN_H
