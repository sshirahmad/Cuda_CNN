#include "../lib/pooling.h"

// CNNLayer Constructor
PoolingLayer::PoolingLayer(cudnnHandle_t cudnn,
                            int inputHeight, int inputWidth,
                            int filterHeight, int filterWidth,
                            int strideHeight, int strideWidth,
                            int paddingHeight, int paddingWidth,
                            int inputChannels, int batchSize)
                :
    cudnn(cudnn),  
    inputHeight(inputHeight), inputWidth(inputWidth),
    filterHeight(filterHeight), filterWidth(filterWidth),
    strideHeight(strideHeight), strideWidth(strideWidth),
    paddingHeight(paddingHeight), paddingWidth(paddingWidth),
    inputChannels(inputChannels), batchSize(batchSize) {
    
    // Initialize and set tensor and convolution descriptors
    CreateandSetDescs();

}

// Destructor
PoolingLayer::~PoolingLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void PoolingLayer::CreateandSetDescs() {

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// FORWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));

    // Pooling descriptor for max pooling
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                            filterHeight, filterWidth, paddingHeight, paddingWidth, strideHeight, strideWidth));


    // Pooling tensor dimensions
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, inputDesc,
                                                  &batchSize, &inputChannels, &poolHeight, &poolWidth));

    // Output pooling tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, poolHeight, poolWidth));

    // Allocate memory for pooling tensor
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * inputChannels * poolHeight * poolWidth * sizeof(float)));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Allocate memory for grad of pooling input tensor
    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float)));

}

// Free GPU memory
void PoolingLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyPoolingDescriptor(poolDesc);

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceInputGrad));
}

// Forward pass
float* PoolingLayer::ForwardPass(const float* deviceInput) {

    this->deviceInput = deviceInput;

    LaunchMaxPoolingKernel();

    return deviceOutput;

}

float* PoolingLayer::BackwardPass(const float* deviceOutputGrad) {

    this->deviceOutputGrad = deviceOutputGrad;

    LaunchBackwardMaxPoolingKernel();

    return deviceInputGrad;

}

void PoolingLayer::LaunchMaxPoolingKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Perform max pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, poolDesc, &alpha, inputDesc, deviceInput,
                                    &beta, outputDesc, deviceOutput));

    cudaDeviceSynchronize();

}

// Backward Max Pooling Kernel
void PoolingLayer::LaunchBackwardMaxPoolingKernel() {

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnPoolingBackward(cudnn, poolDesc, &alpha, outputDesc, deviceOutput,
                                     outputDesc, deviceOutputGrad, inputDesc, deviceInput, &beta, inputDesc, deviceInputGrad));
                                     
    cudaDeviceSynchronize();
}

