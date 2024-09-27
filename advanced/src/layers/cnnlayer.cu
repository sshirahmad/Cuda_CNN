#include "../lib/cnnlayer.h"

// CNNLayer Constructor
CNNLayer::CNNLayer(cudnnHandle_t cudnn, int inputHeight, int inputWidth,
                    int filterHeight, int filterWidth,
                    int strideHeight, int strideWidth,
                    int paddingHeight, int paddingWidth,
                    int outputChannels, int inputChannels,
                    int batchSize, float learningrate)
                :
    cudnn(cudnn),  
    inputHeight(inputHeight), inputWidth(inputWidth),
    filterHeight(filterHeight), filterWidth(filterWidth),
    strideHeight(strideHeight), strideWidth(strideWidth),
    paddingHeight(paddingHeight), paddingWidth(paddingWidth),
    outputChannels(outputChannels), inputChannels(inputChannels),
    batchSize(batchSize), learningrate(learningrate) {
    
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

    // Filter tensor (weights) descriptor (for cudnnAddTensor)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&filterTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(filterTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           outputChannels, inputChannels, filterHeight, filterWidth));

    // Allocate memory for filter tensor
    CHECK_CUDA(cudaMalloc(&deviceFilter, outputChannels * inputChannels * filterHeight * filterWidth * sizeof(float)));

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
    CHECK_CUDA(cudaMalloc(&deviceConv, batchSize * outputChannels * convHeight * convWidth * sizeof(float)));

    ///////////////////////////// ACTIVATION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Activation (ReLU) descriptor
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Allocate memory for activation tensor
    CHECK_CUDA(cudaMalloc(&deviceAct, batchSize * outputChannels * convHeight * convWidth * sizeof(float)));

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
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * outputChannels * poolHeight * poolWidth * sizeof(float)));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    ///////////////////////////// POOLING TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of pooling input tensor
    CHECK_CUDA(cudaMalloc(&deviceActGrad, batchSize * outputChannels * convHeight * convWidth * sizeof(float)));

    ///////////////////////////// ACTIVATION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of activation input tensor
    CHECK_CUDA(cudaMalloc(&deviceConvGrad, batchSize * outputChannels * convHeight * convWidth * sizeof(float)));

    ///////////////////////////// CONVOLUTION TENSORS AND DESCRIPTORS /////////////////////////////   

    // Allocate memory for grad of convolution input tensor
    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float)));

    // Allocate memory for grad of convolution filter tensor
    CHECK_CUDA(cudaMalloc(&deviceFilterGrad, inputChannels * outputChannels * filterHeight * filterWidth * sizeof(float)));


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
    CHECK_CUDA(cudaFree(deviceConv));
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceAct));
    CHECK_CUDA(cudaFree(deviceFilter));

    CHECK_CUDA(cudaFree(deviceActGrad));
    CHECK_CUDA(cudaFree(deviceConvGrad));
    CHECK_CUDA(cudaFree(deviceInputGrad));
    CHECK_CUDA(cudaFree(deviceFilterGrad));

}

// Forward pass
float* CNNLayer::ForwardPass(const float* deviceInput) {

    this->deviceInput = deviceInput;

    LaunchConvolutionKernel();
    LaunchActivationKernel();
    LaunchMaxPoolingKernel();

    return deviceOutput;

}

float* CNNLayer::BackwardPass(const float* deviceOutputGrad) {

    this->deviceOutputGrad = deviceOutputGrad;

    LaunchBackwardMaxPoolingKernel();
    LaunchBackwardActivationKernel();
    LaunchBackwardConvolutionKernel();

    UpdateWeights();

    return deviceInputGrad;

}


void CNNLayer::UpdateWeights() {
    // Update weights (filters)
    float alpha = -learningrate; 
    float beta = 1.0f;            

    // Update filters using gradients
    CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, filterTensorDesc, deviceFilterGrad,
                                &beta, filterTensorDesc, deviceFilter));

    cudaDeviceSynchronize();
}

void CNNLayer::LaunchConvolutionKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, deviceInput, filterDesc, deviceFilter,
                                        convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, 
                                        &beta, outputconvDesc, deviceConv));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchActivationKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Apply ReLU activation function
    CHECK_CUDNN(cudnnActivationForward(cudnn, activationDesc, &alpha, outputconvDesc, deviceConv,
                                       &beta, outputconvDesc, deviceAct));

    cudaDeviceSynchronize();

}

void CNNLayer::LaunchMaxPoolingKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Perform max pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, poolDesc, &alpha, outputconvDesc, deviceAct,
                                    &beta, outputpoolDesc, deviceOutput));

    cudaDeviceSynchronize();

}

// Backward Max Pooling Kernel
void CNNLayer::LaunchBackwardMaxPoolingKernel() {

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnPoolingBackward(cudnn, poolDesc, &alpha, outputpoolDesc, deviceOutput,
                                     outputpoolDesc, deviceOutputGrad, outputconvDesc, deviceAct, &beta, outputconvDesc, deviceActGrad));
                                     
    cudaDeviceSynchronize();
}

// Backward Activation Kernel
void CNNLayer::LaunchBackwardActivationKernel() {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnActivationBackward(cudnn, activationDesc, &alpha, outputconvDesc, deviceAct,
                                        outputconvDesc, deviceActGrad, outputconvDesc, deviceConv, &beta, outputconvDesc, deviceConvGrad));

    cudaDeviceSynchronize();
}

// Backward Convolution Kernel
void CNNLayer::LaunchBackwardConvolutionKernel() {
    float alpha = 1.0f, beta = 0.0f;

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
    int threadsPerBlock = 256; // Choose a value that's a power of 2, usually 256 or 512
    int blocksPerGrid = (filter_num_elements + threadsPerBlock - 1) / threadsPerBlock; // Calculate total blocks needed

    initializeWeights<<<blocksPerGrid, threadsPerBlock>>>(deviceFilter, filter_num_elements, 1234ULL, -0.5f, 0.5f);

    // Ensure the kernel is executed correctly
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " in File " << __FILE__
                << " in line " << __LINE__
                << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

}


// Get output from device to host
std::tuple<int, int, float*> CNNLayer::GetOutput(int index) {

    float* output = deviceOutput + index * poolWidth * poolHeight * outputChannels + 0 * poolHeight * poolWidth;
    // float* output = deviceConv + index * outputChannels * convHeight * convWidth + 0 * convHeight * convWidth;

    return {poolWidth, poolHeight, output};
    // return {convWidth, convHeight, output};

}
