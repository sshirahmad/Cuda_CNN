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

    // Allocate memory for flattened cnn output tensor
    CHECK_CUDA(cudaMalloc(&flattenedOutput, flattenedSize * sizeof(float)));

    // Allocate memory for loss tensor
    CHECK_CUDA(cudaMalloc(&deviceLoss, sizeof(float)));

    // Allocate memory for loss gradient tensor
    CHECK_CUDA(cudaMalloc(&deviceOutputGrad, batchSize * numClass * sizeof(float)));



}

std::tuple<int, int> CNN::CalculateDim(int inHeight, int inWidth){

    int tempHeight = (inHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    int tempWidth = (inWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;

    int outHeight = (tempHeight + 2 * paddingHeight - filterHeight) / strideHeight + 1;
    int outWidth = (tempWidth + 2 * paddingWidth - filterWidth) / strideWidth + 1;

    return {outHeight, outWidth};

}


void CNN::BuildModel() {

    // Build first layer
    int cnnChannels1 = numFilter;
    Layer1 = new CNNLayer(cudnn, inputHeight, inputWidth, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels1, inputChannels, batchSize, learningrate);

    // Build Second layer
    int cnnChannels2 = cnnChannels1 / 2;
    auto[inputHeight2, inputWidth2] = CalculateDim(inputHeight, inputWidth);
    Layer2 = new CNNLayer(cudnn, inputHeight2, inputWidth2, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels2, cnnChannels1, batchSize, learningrate);

    // Build Third layer
    int cnnChannels3 = cnnChannels2 / 2;
    auto[inputHeight3, inputWidth3] = CalculateDim(inputHeight2, inputWidth2);
    Layer3 = new CNNLayer(cudnn, inputHeight3, inputWidth3, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, cnnChannels3, cnnChannels2, batchSize, learningrate);

    outputChannels = cnnChannels3;
    std::tie(outputHeight, outputWidth) = CalculateDim(inputHeight3, inputWidth3);

    // Build fourth layer
    flattenedSize = batchSize * outputHeight * outputWidth * outputChannels;
    Layer4 = new FCLayer(cublas, outputHeight * outputWidth * outputChannels, hiddenDim, batchSize, learningrate);

    // Build fifth layer
    Layer5 = new FCLayer(cublas, hiddenDim, numClass, batchSize, learningrate);

}

// Free GPU memory
void CNN::FreeMemory() {
    // Free intermediate buffers
    CHECK_CUDA(cudaFree(deviceInput));
    CHECK_CUDA(cudaFree(flattenedOutput));
    CHECK_CUDA(cudaFree(deviceLoss));
    CHECK_CUDA(cudaFree(deviceLabels));
    CHECK_CUDA(cudaFree(deviceOutputGrad));

    delete Layer1;
    delete Layer2;
    delete Layer3;
    delete Layer4;
    delete Layer5;

}

// Forward pass
float* CNN::ForwardPass(const float* hostInput, const int* hostLabels) {

    CHECK_CUDA(cudaMemcpy(deviceInput, hostInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceLabels, hostLabels, batchSize * sizeof(int), cudaMemcpyHostToDevice));

    // CNN
    auto deviceOutput1 = Layer1->ForwardPass(deviceInput);
    auto deviceOutput2 = Layer2->ForwardPass(deviceOutput1);
    cnnOutput = Layer3->ForwardPass(deviceOutput2);

    // // Launch flattening kernel
    // int blockSize = 256;  
    // int gridSize = (flattenedSize + blockSize - 1) / blockSize;
    // flatten_NCHW<<<gridSize, blockSize>>>(cnnOutput, flattenedOutput, batchSize, outputChannels, outputHeight, outputWidth);

    // // Ensure the kernel is executed correctly
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err)
    //             << " in File " << __FILE__
    //             << " in line " << __LINE__
    //             << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // FC
    // FCs are implemented using cuBLAS, so all amtrices have to be stored in column major order. 
    // flatten_NCHW stores the cnn output in row major order so we'll use the transpose of matrices. 
    auto deviceOutput4 = Layer4->ForwardPass(cnnOutput);
    fcLogits = Layer5->ForwardPass(deviceOutput4);

    cudaDeviceSynchronize();

    return fcLogits;

}

void CNN::BackwardPass() {

    auto deviceInputGrad5 = Layer5->BackwardPass(deviceOutputGrad);
    auto deviceInputGrad4 = Layer4->BackwardPass(deviceInputGrad5);
    auto deviceInputGrad3 = Layer3->BackwardPass(deviceInputGrad4);
    auto deviceInputGrad2 = Layer2->BackwardPass(deviceInputGrad3);
    auto deviceInputGrad = Layer1->BackwardPass(deviceInputGrad2);

    cudaDeviceSynchronize();

}

float* CNN::ComputeLoss() {
    
    // Call a kernel to compute cross-entropy loss
    int blockSize = 256;  // You can tune this
    int gridSize = (flattenedSize + blockSize - 1) / blockSize;

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


// Get output from device to host
std::tuple<int, int, float*> CNN::GetOutput(int index) {

    // Get filter 0 output from batch idx
    float* output = cnnOutput + index * outputChannels * outputHeight * outputWidth + 0 * outputHeight * outputWidth;

    return {outputWidth, outputHeight, output};

}


