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
            sumExp += expf(logits[idx * numClasses + c]); // Subtract maxLogit for numerical stability
        }

        // Loss for the correct class
        float logProbTrueClass = logits[idx * numClasses + label] - logf(sumExp);

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
            sumExp += expf(logits[idx * numClasses + c]);
        }

        // Compute the gradient for each class
        for (int c = 0; c < numClasses; ++c) {
            float softmax_prob = expf(logits[idx * numClasses + c]) / sumExp;

            if (c == label) {
                grad[idx * numClasses + c] = softmax_prob - 1;  // True class: p_j - 1
            } else {
                grad[idx * numClasses + c] = softmax_prob;         // Other classes: p_j
            }

        }
    }
}

__global__ void calculate_accuracy(float* logits, int* labels, float* accuracy, int numClasses, int batchSize) {
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float correct_predictions = 0;

    if (idx < batchSize) {
        // Find the predicted class (maximum logit)
        int predicted_class = 0;
        float max_logit = logits[idx * numClasses];
        
        for (int c = 1; c < numClasses; ++c) {
            if (logits[idx * numClasses + c] > max_logit) {
                max_logit = logits[idx * numClasses + c];
                predicted_class = c;
            }
        }

        // Compare with the actual label
        if (predicted_class == labels[idx]) {
            correct_predictions = 1;
        }
    
        atomicAdd(accuracy, correct_predictions);

    }
}


