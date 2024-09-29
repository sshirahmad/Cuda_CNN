#ifndef ADAM_H
#define ADAM_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>


#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in " \
                      << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                      \
    }


class Adam {
public:
    Adam(int size, float learningRate = 0.001, float weight_decay = 0.0, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-7);

    ~Adam();

    void update(float* weights, float* gradients);

private:
    int size;
    float learningRate;
    float weight_decay;
    float beta1;
    float beta2;
    float epsilon;
    float* m; // First moment vector
    float* v; // Second moment vector
    int t;    // Time step


    void AllocateMemory();
};

#endif // ADAM_H
