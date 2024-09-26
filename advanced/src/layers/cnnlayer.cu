#include "../lib/cnnlayer.h"

// CNNLayer Constructor
CNNLayer::CNNLayer(cudnnHandle_t cudnn, int inputHeight, int inputWidth,
                    int filterHeight, int filterWidth,
                    int strideHeight, int strideWidth,
                    int paddingHeight, int paddingWidth,
                    int outputChannels, int inputChannels,
                    int batchSize)
                :
    cudnn(cudnn),  
    inputHeight(inputHeight), inputWidth(inputWidth),
    filterHeight(filterHeight), filterWidth(filterWidth),
    strideHeight(strideHeight), strideWidth(strideWidth),
    paddingHeight(paddingHeight), paddingWidth(paddingWidth),
    outputChannels(outputChannels), inputChannels(inputChannels),
    batchSize(batchSize) {
    
    // Initialize and set tensor and convolution descriptors
    CreateandSetDescs();
    SetFilters();
}

// Destructor
CNNLayer::~CNNLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void CNNLayer::CreateandSetDescs() {

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// FORWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));
          
    ///////////////////////////// CONVOLUTION TENSORS AND DESCRIPTORS /////////////////////////////   
    // Filter (weights) descriptor
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           outputChannels, inputChannels, filterHeight, filterWidth));

    // Allocate memory for filter tensor
    cudaMalloc(&deviceFilter, outputChannels * inputChannels * filterHeight * filterWidth * sizeof(float));

    // Convolution descriptor
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, paddingHeight, paddingWidth, strideHeight, strideWidth, 1, 1,
                                                CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // Output tensor dimensions
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc,
                                                      &batchSize, &outputChannels, &convHeight, &convWidth));

    // Output convolution tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputconvDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputconvDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outputChannels, convHeight, convWidth));

    
    // Allocate memory for convolution tensor
    cudaMalloc(&deviceConv, batchSize * outputChannels * convHeight * convWidth * sizeof(float));

    ///////////////////////////// ACTIVATION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Activation (ReLU) descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Allocate memory for activation tensor
    cudaMalloc(&deviceAct, batchSize * outputChannels * convHeight * convWidth * sizeof(float));

    ///////////////////////////// POOLING TENSORS AND DESCRIPTORS /////////////////////////////   

    // Pooling descriptor for max pooling
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                            filterHeight, filterWidth, paddingHeight, paddingWidth, strideHeight, strideWidth));


    // Pooling tensor dimensions
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outputconvDesc,
                                                      &batchSize, &outputChannels, &poolHeight, &poolWidth));

    // Output pooling tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputpoolDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputpoolDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outputChannels, poolHeight, poolWidth));

    // Allocate memory for pooling tensor
    cudaMalloc(&deviceOutput, batchSize * outputChannels * poolHeight * poolWidth * sizeof(float));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    ///////////////////////////// POOLING TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of pooling input tensor
    cudaMalloc(&deviceActGrad, batchSize * outputChannels * convHeight * convWidth * sizeof(float));

    ///////////////////////////// ACTIVATION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of activation input tensor
    cudaMalloc(&deviceConvGrad, batchSize * outputChannels * convHeight * convWidth * sizeof(float));

    ///////////////////////////// CONVOLUTION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of convolution input tensor
    cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float));

    // Allocate memory for grad of convolution filter tensor
    cudaMalloc(&deviceFilterGrad, inputChannels * outputChannels * filterHeight * filterWidth * sizeof(float));


}

// Free GPU memory
void CNNLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputconvDesc);
    cudnnDestroyTensorDescriptor(outputpoolDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyPoolingDescriptor(poolDesc);

    // Free intermediate buffers
    cudaFree(deviceConv);
    cudaFree(deviceOutput);
    cudaFree(deviceAct);
    cudaFree(deviceFilter);

    cudaFree(deviceActGrad);
    cudaFree(deviceConvGrad);
    cudaFree(deviceInputGrad);
    cudaFree(deviceFilterGrad);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Forward pass
float* CNNLayer::ForwardPass(const float* Input) {

    // reset memory
    deviceInput = Input;

    cudaMemset(deviceConv, 0, batchSize * convWidth * convHeight * outputChannels * sizeof(float));
    cudaMemset(deviceAct, 0, batchSize * convWidth * convHeight * outputChannels * sizeof(float));
    cudaMemset(deviceOutput, 0, batchSize * poolWidth * poolHeight * outputChannels * sizeof(float));

    LaunchConvolutionKernel();
    LaunchActivationKernel();
    LaunchMaxPoolingKernel();

    return deviceOutput;

}

float* CNNLayer::BackwardPass(const float* OutputGrad) {

    // Reset gradient buffers
    deviceOutputGrad = OutputGrad;

    cudaMemset(deviceActGrad, 0, batchSize * outputChannels * convHeight * convWidth * sizeof(float));
    cudaMemset(deviceConvGrad, 0, batchSize * outputChannels * convHeight * convWidth * sizeof(float));
    cudaMemset(deviceInputGrad, 0, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float));
    cudaMemset(deviceFilterGrad, 0, outputChannels * inputChannels * filterHeight * filterWidth * sizeof(float));

    LaunchBackwardMaxPoolingKernel();
    LaunchBackwardActivationKernel();
    LaunchBackwardConvolutionKernel();

    return deviceInputGrad;

}


// void CNNLayer::UpdateWeights() {
//     // Update weights (filters)
//     float alpha = -learningRate; // Learning rate scaling factor for gradient descent
//     float beta = 1.0f;            // For in-place update

//     // Update filters using gradients
//     CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, filterDesc, deviceGradFilter,
//                                 &beta, filterDesc, deviceFilter));

//     cudaDeviceSynchronize();
// }

void CNNLayer::LaunchConvolutionKernel() {

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, deviceInput, filterDesc, deviceFilter,
                                        convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, 
                                        &beta, outputconvDesc, deviceConv));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchActivationKernel() {

    // Apply ReLU activation function
    CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, outputconvDesc, deviceConv,
                                       &beta, outputconvDesc, deviceAct));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchMaxPoolingKernel() {

    // Perform max pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, poolDesc, &alpha, outputconvDesc, deviceAct,
                                    &beta, outputpoolDesc, deviceOutput));

    cudaDeviceSynchronize();

}

// Backward Max Pooling Kernel
void CNNLayer::LaunchBackwardMaxPoolingKernel() {
    CHECK_CUDNN(cudnnPoolingBackward(cudnn, poolDesc, &alpha, outputpoolDesc, deviceOutput,
                                     outputpoolDesc, deviceOutputGrad, outputconvDesc, deviceAct, &beta, outputconvDesc, deviceActGrad));
                                     
    cudaDeviceSynchronize();
}

// Backward Activation Kernel
void CNNLayer::LaunchBackwardActivationKernel() {
    CHECK_CUDNN(cudnnActivationBackward(cudnn, activationDesc, &alpha, outputconvDesc, deviceAct,
                                        outputconvDesc, deviceActGrad, outputconvDesc, deviceConv, &beta, outputconvDesc, deviceConvGrad));

    cudaDeviceSynchronize();
}

// Backward Convolution Kernel
void CNNLayer::LaunchBackwardConvolutionKernel() {
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, filterDesc, deviceFilter,
                                             outputconvDesc, deviceConvGrad, convDesc,
                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                             nullptr, 0, &beta, inputDesc, deviceInputGrad));

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn, &alpha, inputDesc, deviceInput,
                                               outputconvDesc, deviceConvGrad, convDesc,
                                               CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                               nullptr, 0, &beta, filterDesc, deviceFilterGrad));

    cudaDeviceSynchronize();
}

// Initialize filters 
void CNNLayer::SetFilters() {
    int filter_num_elements = filterHeight * filterWidth * inputChannels * outputChannels;
    initializeWeights<<<1, filter_num_elements>>>(deviceFilter, filter_num_elements, 1234ULL, -0.5f, 0.5f);
}


// void CNNLayer::UpdateWeights(float learningRate) {
//     int filter_num_elements = filterHeight * filterWidth * inputChannels * outputChannels;
//     updateWeightsKernel<<<1, filter_num_elements>>>(deviceFilter, filter_num_elements, deviceFilterGrad, learningRate);
// }


// Get output from device to host
std::tuple<int, int, float*> CNNLayer::GetOutput(int index) {

    float* output = deviceOutput + index * poolWidth * poolHeight * outputChannels + 0 * poolHeight * poolWidth;
    // float* output = deviceConv + index * outputChannels * convHeight * convWidth + 0 * convHeight * convWidth;

    return {poolWidth, poolHeight, output};
    // return {convWidth, convHeight, output};

}
