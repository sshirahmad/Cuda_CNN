#include "../lib/convolution.h"

// CNNLayer Constructor
ConvolutionLayer::ConvolutionLayer(cudnnHandle_t cudnn, cublasHandle_t cublas,
                    int inputHeight, int inputWidth,
                    int filterHeight, int filterWidth,
                    int strideHeight, int strideWidth,
                    int paddingHeight, int paddingWidth,
                    int outputChannels, int inputChannels,
                    int batchSize, float learningrate,
                    float weight_decay)
                :
    cudnn(cudnn), cublas(cublas),
    inputHeight(inputHeight), inputWidth(inputWidth),
    filterHeight(filterHeight), filterWidth(filterWidth),
    strideHeight(strideHeight), strideWidth(strideWidth),
    paddingHeight(paddingHeight), paddingWidth(paddingWidth),
    outputChannels(outputChannels), inputChannels(inputChannels),
    batchSize(batchSize), learningrate(learningrate),
    weight_decay(weight_decay){
    
    // Initialize and set tensor and convolution descriptors
    CreateandSetDescs();
    SetFilters();
}

// Destructor
ConvolutionLayer::~ConvolutionLayer() {
    FreeMemory();
}

// Allocate memory for GPU data
void ConvolutionLayer::CreateandSetDescs() {

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// FORWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Input tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inputChannels, inputHeight, inputWidth));
          
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
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outputChannels, convHeight, convWidth));

    
    // Allocate memory for convolution tensor
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * outputChannels * convHeight * convWidth * sizeof(float)));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// BACKWARD PASS /////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    // Allocate memory for grad of convolution input tensor
    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputChannels * inputWidth * inputHeight * sizeof(float)));

    // Allocate memory for grad of convolution filter tensor
    CHECK_CUDA(cudaMalloc(&deviceFilterGrad, inputChannels * outputChannels * filterHeight * filterWidth * sizeof(float)));

    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// OPTIMIZERS ////////////////////////////////   
    /////////////////////////////////////////////////////////////////////////

    optimizer = new Adam(inputChannels * outputChannels * filterHeight * filterWidth, batchSize, learningrate, weight_decay);


}

// Free GPU memory
void ConvolutionLayer::FreeMemory() {

    // Clean up descriptors
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceFilter));
    CHECK_CUDA(cudaFree(deviceInputGrad));
    CHECK_CUDA(cudaFree(deviceFilterGrad));

    delete optimizer;

}

// Forward pass
float* ConvolutionLayer::ForwardPass(const float* deviceInput) {

    this->deviceInput = deviceInput;

    LaunchConvolutionKernel();

    return deviceOutput;

}

float* ConvolutionLayer::BackwardPass(const float* deviceOutputGrad) {

    this->deviceOutputGrad = deviceOutputGrad;

    LaunchBackwardConvolutionKernel();

    UpdateWeights();

    return deviceInputGrad;

}


void ConvolutionLayer::UpdateWeights() {

    optimizer->update(deviceFilter, deviceFilterGrad);

    cudaDeviceSynchronize();

}

void ConvolutionLayer::LaunchConvolutionKernel() {
    float alpha = 1.0f, beta = 0.0f;

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, deviceInput, filterDesc, deviceFilter,
                                        convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, 
                                        &beta, outputDesc, deviceOutput));

    cudaDeviceSynchronize();

}

// Backward Convolution Kernel
void ConvolutionLayer::LaunchBackwardConvolutionKernel() {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, filterDesc, deviceFilter,
                                             outputDesc, deviceOutputGrad, convDesc,
                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                             nullptr, 0, &beta, inputDesc, deviceInputGrad));

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn, &alpha, inputDesc, deviceInput,
                                               outputDesc, deviceOutputGrad, convDesc,
                                               CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                               nullptr, 0, &beta, filterDesc, deviceFilterGrad));

    cudaDeviceSynchronize();
}

// Initialize filters 
void ConvolutionLayer::SetFilters() {

    int filter_num_elements = filterHeight * filterWidth * inputChannels * outputChannels;
    int threadsPerBlock = 256; // Choose a value that's a power of 2, usually 256 or 512
    int blocksPerGrid = (filter_num_elements + threadsPerBlock - 1) / threadsPerBlock; // Calculate total blocks needed

    initializeUniformWeights<<<blocksPerGrid, threadsPerBlock>>>(deviceFilter, filter_num_elements, 1234ULL, 0.0f, 0.01f);
    // initializeXavierWeights<<<blocksPerGrid, threadsPerBlock>>>(deviceFilter, filter_num_elements, 1234ULL, filterHeight * filterWidth * inputChannels);

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


void ConvolutionLayer::SaveWeights(FILE* file) {

    if (!file) {
        std::cerr << "Invalid file pointer for loading convolutional weights." << std::endl;
        return;
    }

    // Save layer dimensions (input size, output size)
    fwrite(&inputChannels, sizeof(int), 1, file);
    fwrite(&outputChannels, sizeof(int), 1, file);
    fwrite(&filterHeight, sizeof(int), 1, file);
    fwrite(&filterWidth, sizeof(int), 1, file);

    // Determine the size of the weights
    int filter_num_elements = outputChannels * inputChannels * filterHeight * filterWidth;
    
    // Allocate memory on the host (CPU) to store the weights
    std::vector<float> hostFilter(filter_num_elements);

    // Copy weights from device (GPU) to host (CPU)
    CHECK_CUDA(cudaMemcpy(hostFilter.data(), deviceFilter, filter_num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Write the weights to the file
    fwrite(hostFilter.data(), sizeof(float), filter_num_elements, file);

}



void ConvolutionLayer::LoadWeights(FILE* file) {

    if (!file) {
        std::cerr << "Invalid file pointer for loading convolutional weights." << std::endl;
        return;
    }


    // Read layer dimensions (input size, output size)
    int loadedInputChannels, loadedOutputChannels, loadedFilterHeight, loadedFilterWidth;
    fread(&loadedInputChannels, sizeof(int), 1, file);
    fread(&loadedOutputChannels, sizeof(int), 1, file);
    fread(&loadedFilterHeight, sizeof(int), 1, file);
    fread(&loadedFilterWidth, sizeof(int), 1, file);

    // Ensure that dimensions match
    if (loadedInputChannels != inputChannels || loadedOutputChannels != outputChannels || loadedFilterHeight != filterHeight || loadedFilterWidth != filterWidth) {
        std::cerr << "Layer dimensions mismatch while loading weights and biases!" << std::endl;
        fclose(file);
        return;
    }

    // Determine the size of the weights
    int filter_num_elements = outputChannels * inputChannels * filterHeight * filterWidth;

    // Allocate memory on the host (CPU) to store the weights
    std::vector<float> hostFilter(filter_num_elements);

    // Read the weights from the file
    fread(hostFilter.data(), sizeof(float), filter_num_elements, file);

    // Copy the weights from host (CPU) to device (GPU)
    CHECK_CUDA(cudaMemcpy(deviceFilter, hostFilter.data(), filter_num_elements * sizeof(float), cudaMemcpyHostToDevice));

}


