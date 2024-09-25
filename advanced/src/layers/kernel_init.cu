#include "../lib/kernel_init.h"


__global__ void initializeWeights(float* weights, int size, unsigned long long seed, float min, float max) {
    
    // Define the CUDA random state
    curandState state;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Initialize the random state with the seed
    curand_init(seed, idx, 0, &state);

    // Make sure to not go out of bounds
    if (idx < size) {
        // Generate a random float in the range [min, max]
        float randValue = curand_uniform(&state) * (max - min) + min;
        weights[idx] = randValue;
    }
}