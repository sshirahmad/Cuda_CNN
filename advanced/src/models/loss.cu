#include "../lib/loss.h"


__global__ void cross_entropy_loss_with_logits(float* logits, int* labels, float* loss, int numClasses, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batchSize) {
        int label = labels[idx];  // True label for this sample
        float maxLogit = -FLT_MAX;
        float sumExp = 0.0f;

        // Compute the maximum logit for numerical stability
        for (int c = 0; c < numClasses; ++c) {
            maxLogit = fmaxf(maxLogit, logits[idx * numClasses + c]);
        }

        // Compute the log-sum-exp
        for (int c = 0; c < numClasses; ++c) {
            sumExp += expf(logits[idx * numClasses + c] - maxLogit); // Subtract maxLogit for numerical stability
        }

        // Loss for the correct class
        float logProbTrueClass = logits[idx * numClasses + label] - maxLogit - logf(sumExp);

        // Atomic add to accumulate the loss
        atomicAdd(loss, -logProbTrueClass / batchSize);
    }
}


__global__ void cross_entropy_gradient_with_logits(float* logits, int* labels, float* grad, int numClasses, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batchSize) {
        int label = labels[idx];  // True label for this sample
        float maxLogit = -FLT_MAX;
        float sumExp = 0.0f;

        // Compute the maximum logit for numerical stability
        for (int c = 0; c < numClasses; ++c) {
            maxLogit = fmaxf(maxLogit, logits[idx * numClasses + c]);
        }

        // Compute the denominator of softmax (sum of exponentials)
        for (int c = 0; c < numClasses; ++c) {
            sumExp += expf(logits[idx * numClasses + c] - maxLogit);
        }

        // Compute the gradient for each class
        for (int c = 0; c < numClasses; ++c) {
            float softmax_prob = expf(logits[idx * numClasses + c] - maxLogit) / sumExp;

            if (c == label) {
                grad[idx * numClasses + c] = softmax_prob - 1;  // True class: p_j - 1
            } else {
                grad[idx * numClasses + c] = softmax_prob;         // Other classes: p_j
            }

        }
    }
}
