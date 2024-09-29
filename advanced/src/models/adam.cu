#include "../lib/adam.h"


// Kernel to update weights using Adam optimizer
__global__ void adam_update(float* weights, float* gradients, float* m, float* v, int t, float learningRate, float beta1, float beta2, float epsilon, int size, float weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Increment time step
        float m_t = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];
        float v_t = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];
        
        // Update biased first moment estimate
        m[idx] = m_t;
        
        // Update biased second moment estimate
        v[idx] = v_t;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m_t / (1.0f - pow(beta1, t));
        // Compute bias-corrected second moment estimate
        float v_hat = v_t / (1.0f - pow(beta2, t));
        
        // Update weights
        weights[idx] -= learningRate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * weights[idx]) ;
    }
}

// Adam constructor
Adam::Adam(int size, float learningRate, float weight_decay, float beta1, float beta2, float epsilon)
    : size(size), learningRate(learningRate), weight_decay(weight_decay), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {

        AllocateMemory();
}

void Adam::AllocateMemory(){

    // Initialize moment vectors
    CHECK_CUDA(cudaMalloc(&m, sizeof(float) * size));  // Allocate memory for first moment
    CHECK_CUDA(cudaMalloc(&v, sizeof(float) * size));  // Allocate memory for second moment
    CHECK_CUDA(cudaMemset(m, 0, sizeof(float) * size)); // Initialize first moment to zero
    CHECK_CUDA(cudaMemset(v, 0, sizeof(float) * size)); // Initialize second moment to zero 

}

// Destructor
Adam::~Adam() {
    cudaFree(m); // Free first moment vector
    cudaFree(v); // Free second moment vector
}

// Update method
void Adam::update(float* weights, float* gradients) {
    t++;
    
    // Define CUDA kernel launch parameters
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Call the CUDA kernel to update weights
    adam_update<<<gridSize, blockSize>>>(weights, gradients, m, v, t, learningRate, beta1, beta2, epsilon, size, weight_decay);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}


