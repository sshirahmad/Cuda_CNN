#include "../lib/fclayer.h"

// Constructor
FCLayer::FCLayer(cublasHandle_t cublasHandle, int inputSize, int outputSize, int batchSize, float learningRate, float weight_decay)
    : cublasHandle(cublasHandle), inputSize(inputSize), outputSize(outputSize), batchSize(batchSize), learningRate(learningRate), weight_decay(weight_decay) {
    AllocateMemory();
    InitializeWeights();
}

// Destructor
FCLayer::~FCLayer() {
    FreeMemory();
}

// Allocate device memory
void FCLayer::AllocateMemory() {
    CHECK_CUDA(cudaMalloc(&deviceWeight, inputSize * outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceBias, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceOutput, batchSize * outputSize * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&deviceInputGrad, batchSize * inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceWeightGrad, inputSize * outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceBiasGrad, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&ones, batchSize * sizeof(float)));

    CHECK_CUDA(cudaMemset(ones, 1.0f, batchSize * sizeof(float)));

    optimizer_weights = new Adam(inputSize * outputSize, batchSize, learningRate, weight_decay);
    optimizer_bias = new Adam(outputSize, batchSize, learningRate);


}

// Free device memory
void FCLayer::FreeMemory() {
    CHECK_CUDA(cudaFree(deviceWeight));
    CHECK_CUDA(cudaFree(deviceBias));
    CHECK_CUDA(cudaFree(deviceOutput));
    CHECK_CUDA(cudaFree(deviceInputGrad));
    CHECK_CUDA(cudaFree(deviceWeightGrad));
    CHECK_CUDA(cudaFree(deviceBiasGrad));
    CHECK_CUDA(cudaFree(ones));

    delete optimizer_weights;
    delete optimizer_bias;

}

// Forward pass
float* FCLayer::ForwardPass(const float* deviceInput) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    this->deviceInput = deviceInput;

    // Forward pass using cuBLAS
    CHECK_CUBLAS(cublasSgemm(cublasHandle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                outputSize, batchSize, inputSize,  // Note: outputSize x batchSize x inputSize
                &alpha, 
                deviceWeight, outputSize,          // deviceWeight should be of shape (inputSize, outputSize)
                this->deviceInput, inputSize,      // deviceInput (cnnOutput) is of shape (batchSize, inputSize)
                &beta, 
                deviceOutput, outputSize));   

    // Add biases
    CHECK_CUBLAS(cublasSger(cublasHandle,
            outputSize, batchSize,        
            &alpha,
            deviceBias, 1,                      
            ones, 1,                 
            deviceOutput, outputSize));

    
    cudaDeviceSynchronize();


    return deviceOutput; 
}

// Backward pass
float* FCLayer::BackwardPass(const float* deviceOutputGrad) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Assign the deviceOutputGrad pointer
    this->deviceOutputGrad = deviceOutputGrad;

    // Gradient w.r.t input
    CHECK_CUBLAS(cublasSgemm(cublasHandle, 
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    inputSize, batchSize, outputSize,  // Adjusted: inputSize x batchSize x outputSize
                    &alpha, 
                    deviceWeight, outputSize,            // deviceWeight: shape (inputSize, outputSize)
                    this->deviceOutputGrad, outputSize, // deviceOutputGrad: shape (batchSize, outputSize)
                    &beta, 
                    deviceInputGrad, inputSize));        // deviceInputGrad: shape (inputSize, batchSize)

    // Compute gradients with respect to weights (dW)
    CHECK_CUBLAS(cublasSgemm(cublasHandle, 
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    outputSize, inputSize, batchSize,   // Adjusted: outputSize x inputSize x batchSize
                    &alpha, 
                    this->deviceOutputGrad, outputSize,  // deviceOutputGrad: shape (batchSize, outputSize)
                    deviceInput, inputSize,               // deviceInput: shape (batchSize, inputSize)
                    &beta, 
                    deviceWeightGrad, outputSize));       // deviceWeightGrad: shape (inputSize, outputSize)

    // Gradient w.r.t biases (db)
    CHECK_CUBLAS(cublasSgemv(cublasHandle, 
                    CUBLAS_OP_N, 
                    outputSize, batchSize,                // Correct dimensions: outputSize x batchSize
                    &alpha, 
                    this->deviceOutputGrad, outputSize,   // deviceOutputGrad: shape (batchSize, outputSize)
                    ones, 1,                               // `ones`: vector of size (batchSize)
                    &beta, 
                    deviceBiasGrad, 1));                  // deviceBiasGrad: shape (outputSize)

    
    cudaDeviceSynchronize();

    // Update weights and biases
    UpdateWeightsAndBiases();

    return deviceInputGrad; 
}

void FCLayer::UpdateWeightsAndBiases() {

    optimizer_weights->update(deviceWeight, deviceWeightGrad);
    optimizer_bias->update(deviceBias, deviceBiasGrad);

    cudaDeviceSynchronize();

}


// Initialize weights and biases
void FCLayer::InitializeWeights() {

    int weight_num_elements = inputSize * outputSize;
    int threadsPerBlock = 256; // Choose a value that's a power of 2, usually 256 or 512
    int blocksPerGrid = (weight_num_elements + threadsPerBlock - 1) / threadsPerBlock; // Calculate total blocks needed

    initializeUniformWeights<<<blocksPerGrid, threadsPerBlock>>>(deviceWeight, weight_num_elements, 1234ULL, 0.0f, 0.01f);
    // initializeXavierWeights<<<blocksPerGrid, threadsPerBlock>>>(deviceWeight, weight_num_elements, 1234ULL, inputSize);

    int bias_num_elements = outputSize;
    blocksPerGrid = (bias_num_elements + threadsPerBlock - 1) / threadsPerBlock; // Calculate total blocks needed

    initializeBias<<<blocksPerGrid, threadsPerBlock>>>(deviceBias, bias_num_elements, 0.0f);


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


void FCLayer::SaveWeightsAndBiases(FILE* file) {

    if (!file) {
        std::cerr << "Invalid file pointer for loading convolutional weights." << std::endl;
        return;
    }

    // Save layer dimensions (input size, output size)
    fwrite(&inputSize, sizeof(int), 1, file);
    fwrite(&outputSize, sizeof(int), 1, file);

    // Allocate host memory for weights and biases
    std::vector<float> hostWeights(inputSize * outputSize);
    std::vector<float> hostBias(outputSize);

    // Copy data from device (GPU) to host (CPU)
    CHECK_CUDA(cudaMemcpy(hostWeights.data(), deviceWeight, inputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hostBias.data(), deviceBias, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Write weights to file
    fwrite(hostWeights.data(), sizeof(float), inputSize * outputSize, file);

    // Write biases to file
    fwrite(hostBias.data(), sizeof(float), outputSize, file);

}


void FCLayer::LoadWeightsAndBiases(FILE* file) {

    if (!file) {
        std::cerr << "Invalid file pointer for loading convolutional weights." << std::endl;
        return;
    }

    // Read layer dimensions (input size, output size)
    int loadedInputSize, loadedOutputSize;
    fread(&loadedInputSize, sizeof(int), 1, file);
    fread(&loadedOutputSize, sizeof(int), 1, file);

    // Ensure that dimensions match
    if (loadedInputSize != inputSize || loadedOutputSize != outputSize) {
        std::cerr << "Layer dimensions mismatch while loading weights and biases!" << std::endl;
        fclose(file);
        return;
    }

    // Allocate host memory for weights and biases
    std::vector<float> hostWeights(inputSize * outputSize);
    std::vector<float> hostBias(outputSize);

    // Read weights from file
    fread(hostWeights.data(), sizeof(float), inputSize * outputSize, file);

    // Read biases from file
    fread(hostBias.data(), sizeof(float), outputSize, file);

    // Copy data from host (CPU) to device (GPU)
    CHECK_CUDA(cudaMemcpy(deviceWeight, hostWeights.data(), inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceBias, hostBias.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));

}
