#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include <iostream>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <kernel.h>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " \
                      << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                      \
    }

#define CHECK_CUBLAS(call)                                                    \
    {                                                                          \
        cublasStatus_t status = call;                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "CUBLAS error: " << status << " in " << __FILE__    \
                      << " at line " << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                      \
    }


class FCLayer {
public:
    FCLayer(cublasHandle_t cublasHandle, int inputSize, int outputSize, int batchSize, float learningRate);
    ~FCLayer();

    float* ForwardPass(const float* deviceInput);
    float* BackwardPass(const float* deviceOutputGrad);

private:
    cublasHandle_t cublasHandle;

    int inputSize;
    int outputSize;
    int batchSize;
    float learningRate;

    float* ones;
    float* onesm;

    // Device pointers
    const float* deviceInput;
    float* deviceWeight;
    float* deviceBias;
    float* deviceOutput;

    float* deviceInputGrad;
    float* deviceWeightGrad;
    float* deviceBiasGrad;
    const float* deviceOutputGrad;

    void AllocateMemory();
    void FreeMemory();
    void InitializeWeights();
    void UpdateWeightsAndBiases();

};

#endif // FULLY_CONNECTED_LAYER_H
