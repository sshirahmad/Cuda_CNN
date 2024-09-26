#include "../lib/cnn.h"

// CNNLayer Constructor
CNN::CNN(cudnnHandle_t cudnn, int inputHeight, int inputWidth,
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
    AllocateMemory();
    BuildModel();
}

// Destructor
CNN::~CNN() {
    FreeMemory();
}

// Allocate memory for GPU data
void CNN::AllocateMemory() {

    // Allocate memory for input tensor
    cudaMalloc(&deviceInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float));

    // Allocate memory for loss tensor
    cudaMalloc(&deviceLoss, batchSize * sizeof(float));


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
    Layer1 = new CNNLayer(cudnn, inputHeight, inputWidth, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, outputChannels, inputChannels, batchSize);

    // Build Second layer
    auto[inputHeight2, inputWidth2] = CalculateDim(inputHeight, inputWidth);
    Layer2 = new CNNLayer(cudnn, inputHeight2, inputWidth2, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, outputChannels, outputChannels, batchSize);

    // Build Third layer
    auto[inputHeight3, inputWidth3] = CalculateDim(inputHeight2, inputWidth2);
    Layer3 = new CNNLayer(cudnn, inputHeight3, inputWidth3, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, outputChannels, outputChannels, batchSize);


    std::tie(outputHeight, outputWidth) = CalculateDim(inputHeight3, inputWidth3);

}

// Free GPU memory
void CNN::FreeMemory() {
    // Free intermediate buffers
    cudaFree(deviceInput);

    // Free memory for loss
    cudaFree(deviceLoss);

    delete Layer1;
    delete Layer2;
    delete Layer3;

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
float* CNN::ForwardPass(const float* hostInput) {

    // reset memory
    cudaMemset(deviceInput, 0, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float));
    cudaMemcpy(deviceInput, hostInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    auto deviceOutput1 = Layer1->ForwardPass(deviceInput);
    auto deviceOutput2 = Layer2->ForwardPass(deviceOutput1);
    modelOutput = Layer3->ForwardPass(deviceOutput2);

    return modelOutput;

}

float* CNN::BackwardPass(const float* deviceOutputGrad) {

    auto deviceInputGrad3 = Layer3->BackwardPass(deviceOutputGrad);
    auto deviceInputGrad2 = Layer2->BackwardPass(deviceInputGrad3);
    auto deviceInputGrad = Layer1->BackwardPass(deviceInputGrad2);

    return deviceInputGrad;

}

float CNN::ComputeLoss(float* predictedOutput, float* trueLabels, int batchSize, int numClasses) {
    
    // Call a kernel to compute cross-entropy loss
    computeCrossEntropyLoss<<<batchSize, numClasses>>>(predictedOutput, trueLabels, deviceLoss, numClasses);

    // Sum up the loss across all examples
    float hostLoss = 0.0f;
    cudaMemcpy(&hostLoss, deviceLoss, sizeof(float), cudaMemcpyDeviceToHost);
    


    return hostLoss / batchSize;  // Return average loss
}


// Get output from device to host
std::tuple<int, int, float*> CNN::GetOutput(int index) {

    float* output = modelOutput + index * outputChannels * outputHeight * outputWidth + 0 * outputHeight * outputWidth;

    return {outputWidth, outputHeight, output};

}


