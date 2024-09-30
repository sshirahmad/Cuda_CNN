#ifndef LOSS_H
#define LOSS_H


#include <cuda_runtime.h>   // CUDA runtime (cudaMalloc, cudaMemcpy, etc.)
#include <float.h>          // For FLT_MAX (used in finding the max logit for numerical stability)
#include <stdio.h>          // Standard I/O (for error checking or debugging)



__global__ void cross_entropy_loss_with_logits(float* logits, int* labels, float* loss, int numClasses, int batchSize);
__global__ void cross_entropy_gradient_with_logits(float* logits, int* labels, float* grad, int numClasses, int batchSize);
__global__ void calculate_accuracy(float* logits, int* labels, float* accuracy, int numClasses, int batchSize);


#endif // LOSS_H
