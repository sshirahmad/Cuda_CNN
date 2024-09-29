#include "../lib/cnn.h"

// CNNLayer Constructor
CNN::CNN(cudnnHandle_t cudnn, cublasHandle_t cublas,
                    int inputHeight, int inputWidth,
                    int convfilterHeight, int convfilterWidth,
                    int convstrideHeight, int convstrideWidth,
                    int convpaddingHeight, int convpaddingWidth,
                    int poolfilterHeight, int poolfilterWidth,
                    int poolstrideHeight, int poolstrideWidth,
                    int poolpaddingHeight, int poolpaddingWidth,
                    int numFilter, int inputChannels,
                    int hiddenDim, int numClass,
                    int batchSize, float learningrate,
                    float weight_decay, float dropoutProbability)
                :
    cudnn(cudnn), cublas(cublas),
    inputHeight(inputHeight), inputWidth(inputWidth),
    convfilterHeight(convfilterHeight), convfilterWidth(convfilterWidth),
    convstrideHeight(convstrideHeight), convstrideWidth(convstrideWidth),
    convpaddingHeight(convpaddingHeight), convpaddingWidth(convpaddingWidth),
    poolfilterHeight(poolfilterHeight), poolfilterWidth(poolfilterWidth),
    poolstrideHeight(poolstrideHeight), poolstrideWidth(poolstrideWidth),
    poolpaddingHeight(poolpaddingHeight), poolpaddingWidth(poolpaddingWidth),
    numFilter(numFilter), inputChannels(inputChannels),
    hiddenDim(hiddenDim), numClass(numClass),
    batchSize(batchSize), learningrate(learningrate),
    weight_decay(weight_decay), dropoutProbability(dropoutProbability) {
    
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

std::tuple<int, int> CNN::CalculateDim(int inHeight, int inWidth, std::string type){

    int outHeight;
    int outWidth;

    if (type == "conv"){

        outHeight = (inHeight + 2 * convpaddingHeight - convfilterHeight) / convstrideHeight + 1;
        outWidth = (inWidth + 2 * convpaddingWidth - convfilterWidth) / convstrideWidth + 1;

    } else{

        outHeight = (inHeight + 2 * poolpaddingHeight - poolfilterHeight) / poolstrideHeight + 1;
        outWidth = (inWidth + 2 * poolpaddingWidth - poolfilterWidth) / poolstrideWidth + 1;
    }


    return {outHeight, outWidth};

}


void CNN::BuildModel() {

    // Build First CNN
    int cnnChannels1 = numFilter;
    C1 = new ConvolutionLayer(cudnn, cublas, inputHeight, inputWidth, convfilterHeight, convfilterWidth, convstrideHeight, convstrideWidth, convpaddingHeight, convpaddingWidth, cnnChannels1, inputChannels, batchSize, learningrate, weight_decay);
    auto[inputHeight2, inputWidth2] = CalculateDim(inputHeight, inputWidth, "conv");

    A1 = new ActivationLayer(cudnn, inputHeight2, inputWidth2, cnnChannels1, batchSize);

    D1 = new DropoutLayer(cudnn, inputHeight2, inputWidth2, cnnChannels1, batchSize, dropoutProbability);

    P1 = new PoolingLayer(cudnn, inputHeight2, inputWidth2, poolfilterHeight, poolfilterWidth, poolstrideHeight, poolstrideWidth, poolpaddingHeight, poolpaddingWidth, cnnChannels1, batchSize);
    // auto[inputHeight3, inputWidth3] = CalculateDim(inputHeight2, inputWidth2, "pool");

    // // Build Second CNN
    // int cnnChannels2 = cnnChannels1 / 2;
    // C2 = new ConvolutionLayer(cudnn, cublas, inputHeight3, inputWidth3, convfilterHeight, convfilterWidth, convstrideHeight, convstrideWidth, convpaddingHeight, convpaddingWidth, cnnChannels2, cnnChannels1, batchSize, learningrate, weight_decay);
    // auto[inputHeight4, inputWidth4] = CalculateDim(inputHeight3, inputWidth3, "conv");

    // A2 = new ActivationLayer(cudnn, inputHeight4, inputWidth4, cnnChannels2, batchSize);

    // P2 = new PoolingLayer(cudnn, inputHeight4, inputWidth4, poolfilterHeight, poolfilterWidth, poolstrideHeight, poolstrideWidth, poolpaddingHeight, poolpaddingWidth, cnnChannels2, batchSize);
    // auto[inputHeight5, inputWidth5] = CalculateDim(inputHeight4, inputWidth4, "pool");

    // // Build Third CNN
    // int cnnChannels3 = cnnChannels2 / 2;
    // C3 = new ConvolutionLayer(cudnn, cublas, inputHeight5, inputWidth5, convfilterHeight, convfilterWidth, convstrideHeight, convstrideWidth, convpaddingHeight, convpaddingWidth, cnnChannels3, cnnChannels2, batchSize, learningrate, weight_decay);
    // auto[inputHeight6, inputWidth6] = CalculateDim(inputHeight5, inputWidth5, "conv");

    // A3 = new ActivationLayer(cudnn, inputHeight6, inputWidth6, cnnChannels3, batchSize);

    // P3 = new PoolingLayer(cudnn, inputHeight6, inputWidth6, poolfilterHeight, poolfilterWidth, poolstrideHeight, poolstrideWidth, poolpaddingHeight, poolpaddingWidth, cnnChannels3, batchSize);
    
    std::tie(outputHeight, outputWidth) = CalculateDim(inputHeight2, inputWidth2, "pool");
    outputChannels = cnnChannels1;

    // // Build First FC
    // F4 = new FCLayer(cublas, outputHeight * outputChannels * outputWidth, hiddenDim, batchSize, learningrate, weight_decay);
    // A4 = new ActivationLayer(cudnn, 1, 1, hiddenDim, batchSize);

    // Build Second FC
    F5 = new FCLayer(cublas, outputHeight * outputChannels * outputWidth, numClass, batchSize, learningrate, weight_decay);

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
    delete D1;
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
float* CNN::ForwardPass(const float* hostInput, const int* hostLabels, bool training) {

    CHECK_CUDA(cudaMemcpy(deviceInput, hostInput, batchSize * inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceLabels, hostLabels, batchSize * sizeof(int), cudaMemcpyHostToDevice));

    // CNN
    auto output = C1->ForwardPass(deviceInput);
    output = A1->ForwardPass(output);
    output = D1->ForwardPass(output, training);
    output = P1->ForwardPass(output);

    // output = C2->ForwardPass(output);
    // output = A2->ForwardPass(output);
    // output = P2->ForwardPass(output);

    // output = C3->ForwardPass(output);
    // output = A3->ForwardPass(output);
    // output = P3->ForwardPass(output);

    cnnOutput = output;

    // FC
    // FCs are implemented using cuBLAS, so all amtrices have to be stored in column major order. 
    // auto F4Output = F4->ForwardPass(cnnOutput);
    // auto A4Output = A4->ForwardPass(F4Output);
    output = F5->ForwardPass(output);

    fcLogits = output;

    cudaDeviceSynchronize();

    return fcLogits;

}

void CNN::BackwardPass() {

    auto InputGrad = F5->BackwardPass(deviceOutputGrad);

    // InputGrad = A4->BackwardPass(InputGrad);
    // InputGrad = F4->BackwardPass(InputGrad);

    // InputGrad = P3->BackwardPass(InputGrad);
    // InputGrad = A3->BackwardPass(InputGrad);
    // InputGrad = C3->BackwardPass(InputGrad);

    // InputGrad = P2->BackwardPass(InputGrad);
    // InputGrad = A2->BackwardPass(InputGrad);
    // InputGrad = C2->BackwardPass(InputGrad);

    InputGrad = P1->BackwardPass(InputGrad);
    InputGrad = D1->BackwardPass(InputGrad);
    InputGrad = A1->BackwardPass(InputGrad);
    InputGrad = C1->BackwardPass(InputGrad);

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
    // Open the file in binary mode
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        std::cerr << "Failed to open file for saving weights: " << filename << std::endl;
        return;
    }

    C1->SaveWeights(file);
    // C2->SaveWeights(file);
    // C3->SaveWeights(file);
    // F4->SaveWeightsAndBiases(file);
    F5->SaveWeightsAndBiases(file);

    // Close the file after all layers have been saved
    fclose(file);
    std::cout << "Model weights saved to " << filename << std::endl;

}



void CNN::LoadModelWeights(const std::string& filename) {
    // Open the file once in binary mode
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "Failed to open file for loading weights: " << filename << std::endl;
        return;
    }

    C1->LoadWeights(file);
    // C2->LoadWeights(file);
    // C3->LoadWeights(file);
    // F4->LoadWeightsAndBiases(file);
    F5->LoadWeightsAndBiases(file);

    // Close the file after loading all layers
    fclose(file);
    std::cout << "Model weights loaded from " << filename << std::endl;

}

