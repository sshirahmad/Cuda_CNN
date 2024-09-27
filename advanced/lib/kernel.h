#ifndef KERNEL_H
#define KERNEL_H

#include <curand_kernel.h>

__global__ void initializeWeights(float* weights, int size, unsigned long long seed, float min, float max);


__global__ void flatten_NCHW(float* input, float* output, int batchSize, int channels, int height, int width);


#endif // KERNEL_H
