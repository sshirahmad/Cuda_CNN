#include "../lib/kernel.h"


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



__global__ void flatten_NCHW(float* input, float* output, int batchSize, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sizePerImage = channels * height * width;

    if (idx < batchSize * sizePerImage) {
        int batchIdx = idx / sizePerImage;
        int innerIdx = idx % sizePerImage;

        // Flatten: output stores each image as a row
        output[batchIdx * sizePerImage + innerIdx] = input[idx];
    }
}
