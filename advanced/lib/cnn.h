#ifndef CNN_H
#define CNN_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <nppi.h>
#include <cudnn.h>
#include <cnnlayer.h>

// Error checking macro for cuDNN calls
#define CHECK_CUDNN(call) \
    if ((call) != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

class CNN {
public:
    // Constructor
    CNN(cudnnHandle_t cudnn, int inputHeight, int inputWidth,
            int filterHeight, int filterWidth,
            int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth,
            int outputChannels, int inputChannels,
            int batchSize);

    // Destructor
    ~CNN();

    // Forward pass
    float* ForwardPass(const float* hostInput);

    float ComputeLoss(float* predictedOutput, float* trueLabels, int batchSize, int numClasses);

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
    int outputHeight, outputWidth;
    cudnnHandle_t cudnn;

    // CNN Layers
    CNNLayer* Layer1 = nullptr;
    CNNLayer* Layer2 = nullptr;
    CNNLayer* Layer3 = nullptr;

    float* deviceInput;
    float* deviceLoss;
    float* modelOutput;

    void AllocateMemory();
    void FreeMemory();
    void UpdateFilters();
    void BuildModel();
    std::tuple<int, int> CalculateDim(int inHeight, int inWidth);

};

#endif // CNN_H
