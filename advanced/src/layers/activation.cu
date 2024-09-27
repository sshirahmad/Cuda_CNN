#include "../lib/activation.h"

// CNNLayer Constructor
ActivationLayer::ActivationLayer(cudnnHandle_t cudnn,
                    int inputHeight, int inputWidth,
                    int inputChannels, int batchSize)
                :
                cudnn(cudnn),  
                inputHeight(inputHeight), inputWidth(inputWidth),
                inputChannels(inputChannels), batchSize(batchSize) {
    
    // Initialize and set tensor and convolution descriptors
    CreateandSetDescs();
}

// Destructor
ActivationLayer::~ActivationLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void ActivationLayer::CreateandSetDescs() {

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// FORWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));
          
    // Activation (ReLU) descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Allocate memory for activation tensor
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float)));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Allocate memory for grad of activation input tensor
    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float)));

}

// Free GPU memory
void ActivationLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyActivationDescriptor(activationDesc);

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceInputGrad));

}

// Forward pass
float* ActivationLayer::ForwardPass(const float* deviceInput) {

    this->deviceInput = deviceInput;

    LaunchActivationKernel();

    return deviceOutput;

}

float* ActivationLayer::BackwardPass(const float* deviceOutputGrad) {

    this->deviceOutputGrad = deviceOutputGrad;

    LaunchBackwardActivationKernel();

    return deviceInputGrad;

}

void ActivationLayer::LaunchActivationKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Apply ReLU activation function
    CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, deviceInput,
                                       &beta, inputDesc, deviceOutput));

    cudaDeviceSynchronize();

}

// Backward Activation Kernel
void ActivationLayer::LaunchBackwardActivationKernel() {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnActivationBackward(cudnn, activationDesc, &alpha, inputDesc, deviceOutput,
                                        inputDesc, deviceOutputGrad, inputDesc, deviceInput, &beta, inputDesc, deviceInputGrad));

    cudaDeviceSynchronize();
}


