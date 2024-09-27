#include <../lib/cnnlayer.h>
#include <../lib/cnn.h>
#include <../lib/augmentations.h>
#include <../lib/utils.h>
#include <random>

std::tuple<std::vector<float*>, int, int, int, std::vector<std::string>> read_images(const fs::path& directory) {
    std::vector<float*> images;
    std::vector<std::string> basenames;
    int width = 0;
    int height = 0;
    int channels = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {

        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            // Read the image in grayscale or unchanged
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
            std::string basename = entry.path().stem().string(); 

            if (!img.empty()) {
                width = img.cols;
                height = img.rows;
                channels = img.channels();

                // Allocate memory for float image
                size_t img_size = width * height * channels;
                float* image = AllocateHostMemory<float>(img_size * sizeof(float), "pinned");

                // Convert and copy image data to float
                for (size_t i = 0; i < img_size; ++i) {
                    image[i] = static_cast<float>(img.data[i]);
                }

                images.push_back(image);
                basenames.push_back(basename);

            } else {
                std::cerr << "Failed to load image: " << entry.path() << std::endl;
            }
        } else {
            std::cerr << "Entry is not a regular file or not a PNG: " << entry.path() << std::endl;
        }
    }

    return {images, width, height, channels, basenames}; 
}



std::vector<int> readLabelsFromCSV(const std::string& fileName) {
    std::vector<int> labels;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << fileName << std::endl;
        return labels;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                labels.push_back(std::stoi(value));  // Convert the string to an integer label
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid label: " << value << std::endl;
            }
        }
    }

    file.close();
    return labels;
}


std::tuple<std::string, int, int> parseArguments(int argc, char* argv[]) {
    // Initialize default values
    std::string directory = "../data/train/mnist_images";
    int dstWidth = 320;
    int dstHeight = 240;

    // Iterate through command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check for the directory flag
        if (arg == "-d" && i + 1 < argc) {
            directory = argv[++i];
        }

        // Check for the width flag
        else if (arg == "-w" && i + 1 < argc) {
            try {
                dstWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid width value provided. Using default value 320." << std::endl;
            }
        }
        // Check for the height flag
        else if (arg == "-h" && i + 1 < argc) {
            try {
                dstHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid height value provided. Using default value 240." << std::endl;
            }
        }
    }

    std::cout << "Data Path: " << directory << std::endl;
    std::cout << "Width: " << dstWidth << std::endl;
    std::cout << "Height: " << dstHeight << std::endl;


    return {directory, dstWidth, dstHeight};
}


__host__ void convertToUnsignedChar(const float* input, unsigned char* output, int size) {

    float minRange = *std::min_element(input, input + size);
    float maxRange = *std::max_element(input, input + size);

    std::cout << minRange << std::endl;
    std::cout << maxRange << std::endl;

    if (maxRange == minRange) {
        std::fill(output, output + size, 0);  
    } else {
        for (int i = 0; i < size; ++i) {
            unsigned char scaledValue = static_cast<unsigned char>(255.0f * (input[i] - minRange) / (maxRange - minRange));
            output[i] = scaledValue;
        }
    }
}


__host__ void save_image(int outputWidth, int outputHeight, const float* convImage, int numChannels, std::string filename){

    // Calculate size of image
    int output_size = outputWidth * outputHeight * numChannels;
    size_t conv_size = output_size * sizeof(float);

    // Allocate dynamic memory to host image with flot and unsigned char types
    float* h_conv_image = new float[output_size]; // "new" for malloc
    unsigned char* output = new unsigned char[output_size];

    // Copy image to host
    cudaError_t err = cudaMemcpy(h_conv_image, convImage, conv_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in File " << __FILE__
                  << " in line " << __LINE__
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Convert float host image to unsigned char host image (0-255)
    convertToUnsignedChar(h_conv_image, output, output_size);

    // Create an OpenCV matrix for host image to use OpenCV functions
    cv::Mat convMat(outputHeight, outputHeight, CV_MAKETYPE(CV_8U, numChannels), output);

    // Save the image
    std::string outputFileName = "./output/output_" + filename + ".png";
    cv::imwrite(outputFileName, convMat);

    delete[] h_conv_image;
    delete[] output;

}


// Function to print a progress bar
void printProgressBar(int current, int total, int barWidth = 50) {
    float progress = (float)current / total;
    int pos = barWidth * progress;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();  // Ensure the progress bar is updated in real-time
}


int main(int argc, char* argv[]) {

    // Initialize CUDA and cuDNN and cuBLAS
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    auto[directory, dstWidth, dstHeight] = parseArguments(argc, argv);
    
    // Read images and labels
    std::string image_directory = directory + "mnist_images";
    std::string label_directory =  directory + "mnist_labels.csv";
    auto[h_images, srcWidth, srcHeight, numChannels, filenames] = read_images(image_directory);
    auto labels = readLabelsFromCSV(label_directory);

    // Initialize convolution paramters
    int filterHeight = 3, filterWidth = 3; 
    int strideHeight = 1, strideWidth = 1;
    int paddingHeight = 1, paddingWidth = 1;
    int numFilters = 32;
    int hiddenDim = 32, numClass = 10;
    int batchSize = 128;
    float learningrate = 0.0001;
    bool debug = false;
    int epochs = 100;

    // Construct the augmentor
    ImageAugmentation Augmentor(srcWidth, srcHeight, dstWidth, dstHeight, numChannels);

    // Construct the network
    CNN CNNModel(cudnn, cublas, srcHeight, srcWidth, filterHeight, filterWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, numFilters, numChannels, hiddenDim, numClass, batchSize, learningrate);

    // Vectors for batches
    int imageSize = numChannels * dstHeight * dstWidth;
    std::vector<float> lossPerEpoch(epochs);

    std::vector<float*> batch_images;
    std::vector<float> hostInput(batchSize * imageSize);
    std::vector<int> hostLabel;
    std::vector<std::string> batch_filenames;

    for (size_t e = 0; e < epochs; ++e) {

        float epochLoss = 0.0f;

        std::cout << "Epoch " << e + 1 << "/" << epochs << std::endl;

        for (size_t i = 0; i < h_images.size(); ++i) {

            const auto& img = h_images[i];
            const auto& filename = filenames[i]; 
            const auto& label = labels[i]; 

            // Pre-process images
            float* output = Augmentor.augment(img);

            batch_images.push_back(output);
            batch_filenames.push_back(filename);
            hostLabel.push_back(label);

            // When the batch is full, process it
            if (batch_images.size() == batchSize) {

                // Fill the contiguous vector with image data
                for (size_t i = 0; i < batchSize; ++i) {
                    // Copy data from each individual image to the contiguous vector
                    std::copy(batch_images[i], batch_images[i] + imageSize, hostInput.begin() + i * imageSize);
                }

                // Pass the batch to the network
                auto logits = CNNModel.ForwardPass(hostInput.data(), hostLabel.data()); 
                auto deviceLoss = CNNModel.ComputeLoss(); 
                CNNModel.BackwardPass(); 

                // Copy batch loss from device to host
                float batchLoss = 0.0f;
                cudaMemcpy(&batchLoss, deviceLoss, sizeof(float), cudaMemcpyDeviceToHost);

                epochLoss += batchLoss;

                // Update the progress bar
                printProgressBar(i + 1, h_images.size());

                // Print logits for each class in each batch (debug mode)
                if (debug) {

                    std::vector<float> hostLogits(batchSize * numClass);
                    cudaMemcpy(hostLogits.data(), logits, batchSize * numClass * sizeof(float), cudaMemcpyDeviceToHost);

                    for (size_t i = 0; i < batchSize; ++i) {
                        std::cout << "Batch " << i << " logits: ";
                        for (size_t k = 0; k < numClass; ++k) {
                            std::cout << hostLogits[i * numClass + k] << " "; 
                        }
                        std::cout << std::endl; 
                    }

                    // Save output images
                    for (size_t j = 0; j < batch_images.size(); ++j) {
                        auto [outputWidth, outputHeight, outputImage] = CNNModel.GetOutput(j);  
                        save_image(outputWidth, outputHeight, outputImage, 1, batch_filenames[j]);
                    }
                }

                // Clear the batch
                batch_images.clear();
                hostInput.assign(batchSize * imageSize, 0.0f);  // Maintain size of hostInput
                hostLabel.clear();
                batch_filenames.clear();
            }
        }

        // Average the loss over the batches
        lossPerEpoch[e] = epochLoss / (h_images.size() / batchSize);

        // Display final loss after the epoch
        std::cout << std::endl << "Epoch " << e + 1 << " completed. Loss: " << lossPerEpoch[e] << std::endl << std::endl;
    }


    // Cleanup
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);

    return 0;
}
