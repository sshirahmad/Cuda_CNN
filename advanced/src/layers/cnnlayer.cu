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


    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));
          
    // Allocate memory for pooling tensor
    cudaMalloc(&deviceInput, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float));

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

    // Output tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outputChannels, convHeight, convWidth));

    
    // Allocate memory for convolution tensor
    cudaMalloc(&deviceConv, batchSize * outputChannels * convHeight * convWidth * sizeof(float));

    ///////////////////////////// ACTIVATION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Activation (ReLU) descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    ///////////////////////////// POOLING TENSORS AND DESCRIPTORS /////////////////////////////   

    // Pooling descriptor for max pooling
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                            filterHeight, filterWidth, paddingHeight, paddingWidth, strideHeight, strideWidth));


    // Pooling tensor dimensions
    // poolHeight = (convHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    // poolWidth = (convWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outputDesc,
                                                      &batchSize, &outputChannels, &poolHeight, &poolWidth));

    // Allocate memory for pooling tensor
    cudaMalloc(&devicePool, batchSize * outputChannels * poolHeight * poolWidth * sizeof(float));


}

// Free GPU memory
void CNNLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyActivationDescriptor(activationDesc);
    cudnnDestroyPoolingDescriptor(poolDesc);

    // Free intermediate buffers
    cudaFree(deviceInput);
    cudaFree(deviceConv);
    cudaFree(devicePool);

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
void CNNLayer::ForwardPass(float* hostInput) {

    // reset memory
    cudaMemset(deviceInput, 0, batchSize * inputWidth * inputHeight * inputChannels * sizeof(float));
    cudaMemset(deviceConv, 0, batchSize * convWidth * convHeight * outputChannels * sizeof(float));
    cudaMemset(devicePool, 0, batchSize * poolWidth * poolHeight * outputChannels * sizeof(float));

    // Copy the final result to the output array
    cudaError_t err = cudaMemcpy(deviceInput, hostInput, batchSize * inputChannels * inputHeight * inputWidth * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " in File " << __FILE__
                << " in line " << __LINE__
                << std::endl;
        exit(EXIT_FAILURE);
    }

    LaunchConvolutionKernel();
    LaunchActivationKernel();
    LaunchMaxPoolingKernel();

}

void CNNLayer::LaunchConvolutionKernel() {

    // CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc,
    //                                             convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
    //                                             &workspaceSize));

    // std::cout << workspaceSize;

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, deviceInput, filterDesc, deviceFilter,
                                        convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, 
                                        &beta, outputDesc, deviceConv));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchActivationKernel() {

    // Apply ReLU activation function
    CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, outputDesc, deviceConv,
                                       &beta, outputDesc, deviceConv));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchMaxPoolingKernel() {

    // Perform max pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, poolDesc, &alpha, outputDesc, deviceConv,
                                    &beta, outputDesc, devicePool));

    cudaDeviceSynchronize();

}

// Initialize filters 
void CNNLayer::SetFilters() {
    int filter_num_elements = filterHeight * filterWidth * inputChannels * outputChannels;
    initializeWeights<<<1, filter_num_elements>>>(deviceFilter, filter_num_elements, 1234ULL, -0.5f, 0.5f);
}


// Get output from device to host
std::tuple<int, int, float*> CNNLayer::GetOutput(int index) {

    // float* output = devicePool + index * poolWidth * poolHeight * outputChannels + 0 * poolHeight * poolWidth;
    float* output = deviceConv + index * outputChannels * convHeight * convWidth + 0 * convHeight * convWidth;

    // return {poolWidth, poolHeight, output};
    return {convWidth, convHeight, output};

}
