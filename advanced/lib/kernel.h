#ifndef KERNEL_H
#define KERNEL_H

#include <curand_kernel.h>

__global__ void initializeUniformWeights(float* weights, int size, unsigned long long seed, float min, float max);

__global__ void initializeXavierWeights(float* weights, int size, unsigned long long seed, int numInputs);

__global__ void initializeBias(float* biases, int size, float initialValue);

__global__ void flatten_NCHW(float* input, float* output, int batchSize, int channels, int height, int width);


#endif // KERNEL_H
