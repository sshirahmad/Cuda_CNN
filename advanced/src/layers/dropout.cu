#include "../lib/dropout.h"

// CNNLayer Constructor
DropoutLayer::DropoutLayer(cudnnHandle_t cudnn,
                    int inputHeight, int inputWidth,
                    int inputChannels, int batchSize,
                    float dropoutProbability)
                :
                cudnn(cudnn),  
                inputHeight(inputHeight), inputWidth(inputWidth),
                inputChannels(inputChannels), batchSize(batchSize),
                dropoutProbability(dropoutProbability) {
    
    // Initialize and set tensor and convolution descriptors
    CreateandSetDescs();
}

// Destructor
DropoutLayer::~DropoutLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void DropoutLayer::CreateandSetDescs() {

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// FORWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));


    size_t stateSizeInBytes = batchSize * inputChannels * inputHeight * inputWidth * sizeof(float);
    CHECK_CUDA(cudaMalloc(&states, stateSizeInBytes));
          
    // Dropout descriptor
    // Create dropout descriptor
    cudnnCreateDropoutDescriptor(&dropoutDesc);
    cudnnSetDropoutDescriptor(dropoutDesc, 
                                cudnn, 
                                dropoutProbability, 
                                states, 
                                stateSizeInBytes, 
                                0);

    // Allocate memory for dropout output tensor
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float)));

    // Allocate memory for dropout reserve space
    size_t reserveSpaceSize = batchSize * inputChannels * inputHeight * inputWidth * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dropoutMask, reserveSpaceSize));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Allocate memory for grad of activation input tensor
    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float)));

}

// Free GPU memory
void DropoutLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyDropoutDescriptor(dropoutDesc);

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceInputGrad));
    CHECK_CUDA(cudaFree(dropoutMask));
    CHECK_CUDA(cudaFree(states));

}

// Forward pass
float* DropoutLayer::ForwardPass(const float* deviceInput, bool training) {

    this->deviceInput = deviceInput;

    LaunchDropoutKernel(training);

    return deviceOutput;

}

float* DropoutLayer::BackwardPass(const float* deviceOutputGrad) {

    this->deviceOutputGrad = deviceOutputGrad;

    LaunchBackwardDropoutKernel();

    return deviceInputGrad;

}

void DropoutLayer::LaunchDropoutKernel(bool training) {
    float alpha = 1.0f, beta = 0.0f;

    if (training) {
        // Apply dropout during training
        CHECK_CUDNN(cudnnDropoutForward(cudnn, 
                            dropoutDesc, 
                            inputDesc,
                            deviceInput, 
                            inputDesc,
                            deviceOutput, 
                            dropoutMask, 
                            batchSize * inputChannels * inputHeight * inputWidth * sizeof(float)));
    } else {
        // During inference, just copy the input to output
        CHECK_CUDA(cudaMemcpy(deviceOutput, deviceInput, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    cudaDeviceSynchronize();

}

// Backward Activation Kernel
void DropoutLayer::LaunchBackwardDropoutKernel() {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnDropoutBackward(cudnn, dropoutDesc,
                                    inputDesc, deviceOutputGrad,
                                    inputDesc, deviceInputGrad, 
                                    dropoutMask, 
                                    batchSize * inputChannels * inputHeight * inputWidth * sizeof(float)));

    cudaDeviceSynchronize();
}


