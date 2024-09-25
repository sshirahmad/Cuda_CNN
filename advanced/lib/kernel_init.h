#include <curand_kernel.h>

__global__ void initializeWeights(float* weights, int size, unsigned long long seed, float min, float max);