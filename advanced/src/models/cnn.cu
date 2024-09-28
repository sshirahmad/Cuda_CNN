#include "../lib/cnn.h"

// CNNLayer Constructor
CNN::CNN(cudnnHandle_t cudnn, cublasHandle_t cublas,
                    int inputHeight, int inputWidth,
                    int filterHeight, int filterWidth,
                    int strideHeight, int strideWidth,
                    int paddingHeight, int paddingWidth,
                    int numFilter, int inputChannels,
                    int hiddenDim, int numClass,
                    int batchSize, float learningrate)
                :
    cudnn(cudnn), cublas(cublas),
    inputHeight(inputHeight), inputWidth(inputWidth),
    filterHeight(filterHeight), filterWidth(filterWidth),
    strideHeight(strideHeight), strideWidth(strideWidth),
    paddingHeight(paddingHeight), paddingWidth(paddingWidth),
    numFilter(numFilter), inputChannels(inputChannels),
    hiddenDim(hiddenDim), numClass(numClass),
    batchSize(batchSize), learningrate(learningrate) {
    
    // Initialize and set tensor and convolution descriptors
    BuildModel();
    AllocateMemory();
}

// Destructor
CNN::~CNN() {
    FreeMemory();
}

// Allocate memory for GPU data
void CNN::AllocateMemory() {

    // Allocate memory for input tensor
    CHECK_CUDA(cudaMalloc(&deviceInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&deviceLabels, batchSize * sizeof(int)));

    // Allocate memory for loss tensor
    CHECK_CUDA(cudaMalloc(&deviceLoss, sizeof(float)));

    // Allocate memory for accuracy tensor
    CHECK_CUDA(cudaMalloc(&deviceAccuracy, sizeof(float)));

    // Allocate memory for loss gradient tensor
    CHECK_CUDA(cudaMalloc(&deviceOutputGrad, batchSize * numClass * sizeof(float)));

}

std::tuple<int, int> CNN::CalculateDim(int inHeight, int inWidth){

    int outHeight = (inHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    int outWidth = (inWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;

    return {outHeight, outWidth};

}


void CNN::BuildModel() {

    // Build First CNN
    int cnnChannels1 = numFilter;
    C1 = new ConvolutionLayer(cudnn, cublas, inputHeight, inputWidth, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels1, inputChannels, batchSize, learningrate);
    auto[inputHeight2, inputWidth2] = CalculateDim(inputHeight, inputWidth);

    A1 = new ActivationLayer(cudnn, inputHeight2, inputWidth2, cnnChannels1, batchSize);

    P1 = new PoolingLayer(cudnn, inputHeight2, inputWidth2, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels1, batchSize);
    auto[inputHeight3, inputWidth3] = CalculateDim(inputHeight2, inputWidth2);

    // Build Second CNN
    int cnnChannels2 = cnnChannels1 / 2;
    C2 = new ConvolutionLayer(cudnn, cublas, inputHeight3, inputWidth3, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels2, cnnChannels1, batchSize, learningrate);
    auto[inputHeight4, inputWidth4] = CalculateDim(inputHeight3, inputWidth3);

    A2 = new ActivationLayer(cudnn, inputHeight4, inputWidth4, cnnChannels2, batchSize);

    P2 = new PoolingLayer(cudnn, inputHeight4, inputWidth4, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels2, batchSize);
    auto[inputHeight5, inputWidth5] = CalculateDim(inputHeight4, inputWidth4);

    // Build Third CNN
    int cnnChannels3 = cnnChannels2 / 2;
    C3 = new ConvolutionLayer(cudnn, cublas, inputHeight5, inputWidth5, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels3, cnnChannels2, batchSize, learningrate);
    auto[inputHeight6, inputWidth6] = CalculateDim(inputHeight5, inputWidth5);

    A3 = new ActivationLayer(cudnn, inputHeight6, inputWidth6, cnnChannels3, batchSize);

    P3 = new PoolingLayer(cudnn, inputHeight6, inputWidth6, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels3, batchSize);
    std::tie(outputHeight, outputWidth) = CalculateDim(inputHeight6, inputWidth6);
    outputChannels = cnnChannels3;

    // Build First FC
    // flattenedSize = batchSize * outputHeight * outputWidth * outputChannels;
    F4 = new FCLayer(cublas, outputHeight * outputChannels * outputWidth, hiddenDim, batchSize, learningrate);
    A4 = new ActivationLayer(cudnn, 1, 1, hiddenDim, batchSize);

    // Build Second FC
    F5 = new FCLayer(cublas, hiddenDim, numClass, batchSize, learningrate);

}

// Free GPU memory
void CNN::FreeMemory() {
    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceInput));
    CHECK_CUDA(cudaFree(deviceLoss));
    CHECK_CUDA(cudaFree(deviceAccuracy));
    CHECK_CUDA(cudaFree(deviceLabels));
    CHECK_CUDA(cudaFree(deviceOutputGrad));

    delete C1;
    delete A1;
    delete P1;
    delete C2;
    delete A2;
    delete P2;
    delete C3;
    delete A3;
    delete P3;
    delete F4;
    delete A4;
    delete F5;

}

// Forward pass
float* CNN::ForwardPass(const float* hostInput, const int* hostLabels) {

    CHECK_CUDA(cudaMemcpy(deviceInput, hostInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceLabels, hostLabels, batchSize * sizeof(int), cudaMemcpyHostToDevice));

    // CNN
    auto C1Output = C1->ForwardPass(deviceInput);
    auto A1Output = A1->ForwardPass(C1Output);
    auto P1Output = P1->ForwardPass(A1Output);

    auto C2Output = C2->ForwardPass(P1Output);
    auto A2Output = A2->ForwardPass(C2Output);
    auto P2Output = P2->ForwardPass(A2Output);

    auto C3Output = C3->ForwardPass(P2Output);
    auto A3Output = A3->ForwardPass(C3Output);
    auto P3Output = P3->ForwardPass(A3Output);

    cnnOutput = P3Output;

    // FC
    // FCs are implemented using cuBLAS, so all amtrices have to be stored in column major order. 
    auto F4Output = F4->ForwardPass(cnnOutput);
    auto A4Output = A4->ForwardPass(F4Output);
    auto F5Output = F5->ForwardPass(A4Output);

    fcLogits = F5Output;

    cudaDeviceSynchronize();

    return fcLogits;

}

void CNN::BackwardPass() {

    auto F5InputGrad = F5->BackwardPass(deviceOutputGrad);

    auto A4InputGrad = A4->BackwardPass(F5InputGrad);
    auto F4InputGrad = F4->BackwardPass(A4InputGrad);

    auto P3InputGrad = P3->BackwardPass(F4InputGrad);
    auto A3InputGrad = A3->BackwardPass(P3InputGrad);
    auto C3InputGrad = C3->BackwardPass(A3InputGrad);

    auto P2InputGrad = P2->BackwardPass(C3InputGrad);
    auto A2InputGrad = A2->BackwardPass(P2InputGrad);
    auto C2InputGrad = C2->BackwardPass(A2InputGrad);

    auto P1InputGrad = P1->BackwardPass(C2InputGrad);
    auto A1InputGrad = A1->BackwardPass(P1InputGrad);
    auto C1InputGrad = C1->BackwardPass(A1InputGrad);

    cudaDeviceSynchronize();

}

float* CNN::ComputeLoss() {

    CHECK_CUDA(cudaMemset(deviceLoss, 0.0f, sizeof(float)));
    
    // Call a kernel to compute cross-entropy loss
    int blockSize = 256;  // You can tune this
    int gridSize = (batchSize + blockSize - 1) / blockSize;

    cross_entropy_loss_with_logits<<<gridSize, blockSize>>>(fcLogits, deviceLabels, deviceLoss, numClass, batchSize);

    cross_entropy_gradient_with_logits<<<gridSize, blockSize>>>(fcLogits, deviceLabels, deviceOutputGrad, numClass, batchSize);

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

    return deviceLoss;
}


float* CNN::ComputeAccuracy() {

    CHECK_CUDA(cudaMemset(deviceAccuracy, 0.0f, sizeof(float)));

    // Call a kernel to compute cross-entropy loss
    int blockSize = 256;  // You can tune this
    int gridSize = (batchSize + blockSize - 1) / blockSize;

    calculate_accuracy<<<gridSize, blockSize>>>(fcLogits, deviceLabels, deviceAccuracy, numClass, batchSize);

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

    return deviceAccuracy;
}

// Get output from device to host
std::tuple<int, int, float*> CNN::GetOutput(int index) {

    // Get filter 0 output from batch idx
    float* output = cnnOutput + index * outputChannels * outputHeight * outputWidth + 0 * outputHeight * outputWidth;

    return {outputWidth, outputHeight, output};

}



void CNN::SaveModelWeights(const std::string& filename) {

    C1->SaveWeights(filename);
    C2->SaveWeights(filename);
    C3->SaveWeights(filename);
    F4->SaveWeightsAndBiases(filename);
    F5->SaveWeightsAndBiases(filename);

    std::cout << "Model weights saved to " << filename << std::endl;

}



void CNN::LoadModelWeights(const std::string& filename) {

    C1->LoadWeights(filename);
    C2->LoadWeights(filename);
    C3->LoadWeights(filename);
    F4->LoadWeightsAndBiases(filename);
    F5->LoadWeightsAndBiases(filename);

    std::cout << "Model weights loaded from " << filename << std::endl;

}

