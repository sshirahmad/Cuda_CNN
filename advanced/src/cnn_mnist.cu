#include <../lib/cnn.h>
#include <../lib/augmentations.h>
#include <../lib/utils.h>
#include <random>
#include <algorithm> 
#include <numeric> 

std::tuple<std::vector<float*>, int, int, int, std::vector<std::string>> read_images(const fs::path& directory) {
    std::vector<float*> images;
    std::vector<std::string> basenames;
    int width = 0;
    int height = 0;
    int channels = 0;

    // Sort the filenames and make sure the labels and images match. Important step. If not done properly the network will always overfit
    // Vector to store directory entries
    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            entries.push_back(entry);
        }
    }

    // Custom comparator to sort based on the numeric value of the filename stem (basename)
    std::sort(entries.begin(), entries.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
        int num_a = std::stoi(a.path().stem().string());  // Convert basename to an integer
        int num_b = std::stoi(b.path().stem().string());
        return num_a < num_b;
    });

    // Collect all regular PNG files
    for (const auto& entry : entries) {

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


struct Arguments {
    std::string directory;
    std::string save_directory;
    std::string load_directory;
    int dstWidth;
    int dstHeight;
    int convfilterHeight;
    int convfilterWidth;
    int convstrideHeight;
    int convstrideWidth;
    int convpaddingHeight;
    int convpaddingWidth;
    int poolfilterHeight;
    int poolfilterWidth;
    int poolstrideHeight;
    int poolstrideWidth;
    int poolpaddingHeight;
    int poolpaddingWidth;
    int numFilters;
    int hiddenDim;
    int numClass;
    int batchSize;
    float learningRate;
    bool debug;
    int epochs;
    float weight_decay;
    float dropoutProbability;
};

Arguments parseArguments(int argc, char* argv[]) {
    // Initialize default values
    Arguments args = {
        "../data/", // directory
        "./output/weights/", // Save directory
        "./output/weights/weights_best.bin", // Load directory
        28,                          // dstWidth
        28,                          // dstHeight
        3,                            // convfilterHeight
        3,                            // convfilterWidth
        1,                            // convstrideHeight
        1,                            // convstrideWidth
        1,                            // convpaddingHeight
        1,                            // convpaddingWidth
        2,                            // poolfilterHeight
        2,                            // poolfilterWidth
        2,                            // poolstrideHeight
        2,                            // poolstrideWidth
        0,                            // poolpaddingHeight
        0,                            // poolpaddingWidth
        16,                           // numFilters
        128,                           // hiddenDim
        10,                           // numClass
        128,                          // batchSize
        0.0001f,                     // learningRate
        false,                       // debug
        200,                         // epochs
        0.001f,                     // weight_decay
        0.5                         // dropoutProbability                 
    };

    // Iterate through command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check for the directory flag
        if (arg == "-d" && i + 1 < argc) {
            args.directory = argv[++i];
        }

        // Check for the save directory flag
        else if (arg == "-ds" && i + 1 < argc) {
            args.save_directory = argv[++i];
        }

        // Check for the load directory flag
        else if (arg == "-dl" && i + 1 < argc) {
            args.load_directory = argv[++i];
        }

        // Check for the width flag
        else if (arg == "-w" && i + 1 < argc) {
            try {
                args.dstWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid width value provided. Using default value 320." << std::endl;
            }
        }
        // Check for the height flag
        else if (arg == "-h" && i + 1 < argc) {
            try {
                args.dstHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid height value provided. Using default value 240." << std::endl;
            }
        }
        // Check for convolution filter height
        else if (arg == "-cfh" && i + 1 < argc) {
            try {
                args.convfilterHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution filter height value provided. Using default value 3." << std::endl;
            }
        }
        // Check for convolution filter width
        else if (arg == "-cfw" && i + 1 < argc) {
            try {
                args.convfilterWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution filter width value provided. Using default value 3." << std::endl;
            }
        }
        // Check for convolution stride height
        else if (arg == "-csh" && i + 1 < argc) {
            try {
                args.convstrideHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution stride height value provided. Using default value 1." << std::endl;
            }
        }
        // Check for convolution stride width
        else if (arg == "-csw" && i + 1 < argc) {
            try {
                args.convstrideWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution stride width value provided. Using default value 1." << std::endl;
            }
        }
        // Check for convolution padding height
        else if (arg == "-cph" && i + 1 < argc) {
            try {
                args.convpaddingHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution padding height value provided. Using default value 1." << std::endl;
            }
        }
        // Check for convolution padding width
        else if (arg == "-cpw" && i + 1 < argc) {
            try {
                args.convpaddingWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid convolution padding width value provided. Using default value 1." << std::endl;
            }
        }
        // Check for pooling filter height
        else if (arg == "-pfh" && i + 1 < argc) {
            try {
                args.poolfilterHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling filter height value provided. Using default value 2." << std::endl;
            }
        }
        // Check for pooling filter width
        else if (arg == "-pfw" && i + 1 < argc) {
            try {
                args.poolfilterWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling filter width value provided. Using default value 2." << std::endl;
            }
        }
        // Check for pooling stride height
        else if (arg == "-psh" && i + 1 < argc) {
            try {
                args.poolstrideHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling stride height value provided. Using default value 2." << std::endl;
            }
        }
        // Check for pooling stride width
        else if (arg == "-psw" && i + 1 < argc) {
            try {
                args.poolstrideWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling stride width value provided. Using default value 2." << std::endl;
            }
        }
        // Check for pooling padding height
        else if (arg == "-pph" && i + 1 < argc) {
            try {
                args.poolpaddingHeight = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling padding height value provided. Using default value 0." << std::endl;
            }
        }
        // Check for pooling padding width
        else if (arg == "-ppw" && i + 1 < argc) {
            try {
                args.poolpaddingWidth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid pooling padding width value provided. Using default value 0." << std::endl;
            }
        }
        // Check for number of filters
        else if (arg == "-nf" && i + 1 < argc) {
            try {
                args.numFilters = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number of filters provided. Using default value 64." << std::endl;
            }
        }
        // Check for hidden dimension
        else if (arg == "-hd" && i + 1 < argc) {
            try {
                args.hiddenDim = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid hidden dimension value provided. Using default value 64." << std::endl;
            }
        }
        // Check for number of classes
        else if (arg == "-nc" && i + 1 < argc) {
            try {
                args.numClass = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number of classes provided. Using default value 10." << std::endl;
            }
        }
        // Check for batch size
        else if (arg == "-bs" && i + 1 < argc) {
            try {
                args.batchSize = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid batch size value provided. Using default value 128." << std::endl;
            }
        }
        // Check for learning rate
        else if (arg == "-lr" && i + 1 < argc) {
            try {
                args.learningRate = std::stof(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid learning rate value provided. Using default value 0.0001." << std::endl;
            }
        }
        // Check for debug flag
        else if (arg == "-debug") {
            args.debug = true;
        }
        // Check for epochs
        else if (arg == "-e" && i + 1 < argc) {
            try {
                args.epochs = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid epochs value provided. Using default value 100." << std::endl;
            }
        }
        // Check for weight decay
        else if (arg == "-wd" && i + 1 < argc) {
            try {
                args.weight_decay = std::stof(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid weight decay value provided. Using default value 0.001." << std::endl;
            }
        }
        // Check for dropout probability
        else if (arg == "-dp" && i + 1 < argc) {
            try {
                args.dropoutProbability = std::stof(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid dropout probability value provided. Using default value 0.5." << std::endl;
            }
        }
    }

    // Print all parameters
    std::cout << "Data Path: " << args.directory << std::endl;
    std::cout << "Save Path: " << args.save_directory << std::endl;
    std::cout << "Load Path: " << args.load_directory << std::endl;
    std::cout << "Width: " << args.dstWidth << std::endl;
    std::cout << "Height: " << args.dstHeight << std::endl;
    std::cout << "Convolution filter Height: " << args.convfilterHeight << std::endl;
    std::cout << "Convolution filter Width: " << args.convfilterWidth << std::endl;
    std::cout << "Convolution stride Height: " << args.convstrideHeight << std::endl;
    std::cout << "Convolution stride Width: " << args.convstrideWidth << std::endl;
    std::cout << "Convolution padding Height: " << args.convpaddingHeight << std::endl;
    std::cout << "Convolution padding Width: " << args.convpaddingWidth << std::endl;
    std::cout << "Pooling filter Height: " << args.poolfilterHeight << std::endl;
    std::cout << "Pooling filter Width: " << args.poolfilterWidth << std::endl;
    std::cout << "Pooling stride Height: " << args.poolstrideHeight << std::endl;
    std::cout << "Pooling stride Width: " << args.poolstrideWidth << std::endl;
    std::cout << "Pooling padding Height: " << args.poolpaddingHeight << std::endl;
    std::cout << "Pooling padding Width: " << args.poolpaddingWidth << std::endl;
    std::cout << "Number of Filters: " << args.numFilters << std::endl;
    std::cout << "Hidden Dimension: " << args.hiddenDim << std::endl;
    std::cout << "Number of Classes: " << args.numClass << std::endl;
    std::cout << "Batch Size: " << args.batchSize << std::endl;
    std::cout << "Learning Rate: " << args.learningRate << std::endl;
    std::cout << "Debug Mode: " << (args.debug ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Epochs: " << args.epochs << std::endl;
    std::cout << "Weight decay: " << args.weight_decay << std::endl;
    std::cout << "Dropout probability: " << args.dropoutProbability << std::endl;

    return args;
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


float test(std::vector<float*> test_h_images, std::vector<int> test_labels, CNN& CNNModel, ImageAugmentation& Augmentor,
            int numChannels, int dstHeight, int dstWidth, int batchSize){

    //////////////////////// TEST ////////////////////////


    float accuracy = 0.0f;
    int imageSize = numChannels * dstHeight * dstWidth;
    size_t totalSamples = test_h_images.size();
    size_t numBatches = (totalSamples + batchSize - 1) / batchSize;  // Calculate number of batches
    std::vector<float*> batch_images_test;
    std::vector<int> batch_labels_test;
    std::vector<float> hostInput(batchSize * imageSize);  // Buffer for batch input
    for (size_t batch = 0; batch < numBatches; ++batch) {

        // Collect images and labels for the current batch
        for (size_t i = 0; i < batchSize; ++i) {
            size_t idx = batch * batchSize + i;
            if (idx >= totalSamples) break;  // Avoid overflow for the last batch

            const auto& img = test_h_images[idx];
            const auto& label = test_labels[idx];

            // Pre-process images
            float* output = Augmentor.augment(img);
            batch_images_test.push_back(output);
            batch_labels_test.push_back(label);
        }

        // Ensure that the input buffer is properly prepared for the current batch
        for (size_t i = 0; i < batch_images_test.size(); ++i) {
            std::copy(batch_images_test[i], batch_images_test[i] + imageSize, hostInput.begin() + i * imageSize);
        }

        // Forward pass for the current batch
        auto logits = CNNModel.ForwardPass(hostInput.data(), batch_labels_test.data(), false); 

        // Compute accuracy for the current batch
        auto deviceAccuracy = CNNModel.ComputeAccuracy();  

        // Copy batch accuracy from device to host
        float batchAccuracy = 0.0f;
        cudaMemcpy(&batchAccuracy, deviceAccuracy, sizeof(float), cudaMemcpyDeviceToHost);

        accuracy += batchAccuracy;

        // Free batch-specific memory if necessary
        for (auto& img : batch_images_test) {
            delete[] img;  // Free memory for each augmented image
        }
        batch_images_test.clear();
        batch_labels_test.clear();
        hostInput.assign(batchSize * imageSize, 0.0f);  

    }

    // Normalize by the total number of samples to get overall accuracy
    accuracy /= totalSamples;

    std::cout << "Overall Accuracy: " << accuracy * 100.0f << "%" << std::endl;

    return accuracy;

}

void train(std::vector<float*> train_h_images, std::vector<int> train_labels, std::vector<float*> test_h_images, std::vector<int> test_labels,
             std::vector<std::string> train_filenames, CNN& CNNModel, ImageAugmentation& Augmentor,
             int numChannels, int dstHeight, int dstWidth, int epochs, int batchSize, int numClass, bool debug, std::string save_directory){

    //////////////////////// TRAINING ////////////////////////

    // Vectors for batches
    int imageSize = numChannels * dstHeight * dstWidth;
    std::vector<float> lossPerEpoch(epochs);
    std::vector<float*> batch_images;
    std::vector<float> hostInput(batchSize * imageSize);
    std::vector<int> hostLabel;
    std::vector<std::string> batch_filenames;
    float best_accuracy = 0.0f;
    for (size_t e = 0; e < epochs; ++e) {

        // Shuffle data at the start of each epoch
        std::vector<size_t> indices(train_h_images.size());
        std::iota(indices.begin(), indices.end(), 0); // Create an index array [0, 1, 2, ..., n-1]

        // Use a random engine for shuffling
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g); 

        // Shuffle images, filenames, and labels based on the shuffled indices
        std::vector<std::string> shuffledFilenames(train_filenames.size());
        std::vector<float*> shuffledImages(train_h_images.size());
        std::vector<int> shuffledLabels(train_labels.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            shuffledImages[i] = train_h_images[indices[i]];
            shuffledFilenames[i] = train_filenames[indices[i]];
            shuffledLabels[i] = train_labels[indices[i]];
        }

        float epochLoss = 0.0f;

        std::cout << "Epoch " << e + 1 << "/" << epochs << std::endl;

        for (size_t i = 0; i < shuffledImages.size(); ++i) {

            const auto& img = shuffledImages[i];
            const auto& filename = shuffledFilenames[i]; 
            const auto& label = shuffledLabels[i]; 

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
                auto logits = CNNModel.ForwardPass(hostInput.data(), hostLabel.data(), true); 
                auto deviceLoss = CNNModel.ComputeLoss(); 
                CNNModel.BackwardPass(); 

                // Copy batch loss from device to host
                float batchLoss = 0.0f;
                cudaMemcpy(&batchLoss, deviceLoss, sizeof(float), cudaMemcpyDeviceToHost);

                epochLoss += batchLoss;

                // Update the progress bar
                printProgressBar(i + 1, train_h_images.size());

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
                // There will be memory leak if memories are not freed separately
                for (auto& img : batch_images) {
                    delete[] img;  // Free memory for each augmented image
                }
                batch_images.clear();
                hostInput.assign(batchSize * imageSize, 0.0f);  // Maintain size of hostInput
                hostLabel.clear();
                batch_filenames.clear();
            }
        }

        // save weights and validate every ten epochs
        if (e % 10 == 9){
            // Create a directory
            std::filesystem::path dirPath(save_directory);
            
            if (std::filesystem::create_directory(dirPath)) {
                std::cout << "Directory created: " << save_directory << std::endl;
            } else {
                std::cout << "Directory already exists or failed to create: " << save_directory << std::endl;
            }

            // Validation
            auto accuracy = test(test_h_images, test_labels, CNNModel, Augmentor, numChannels, dstHeight, dstWidth, batchSize);

            if (best_accuracy < accuracy){
                best_accuracy = accuracy;
                std::string best_filename = save_directory + "weights_best.bin";
                CNNModel.SaveModelWeights(best_filename);

            }

            std::string filename = save_directory + "weights_" + std::to_string(e + 1) + ".bin";
            CNNModel.SaveModelWeights(filename);

        }

        // Average the loss over the batches
        lossPerEpoch[e] = epochLoss / (train_h_images.size() / batchSize);

        // Display final loss after the epoch
        std::cout << std::endl << "Epoch " << e + 1 << " completed. Loss: " << lossPerEpoch[e] << std::endl << std::endl;
    }

}

int main(int argc, char* argv[]) {

    // Initialize CUDA and cuDNN and cuBLAS
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    Arguments args = parseArguments(argc, argv);
    
    // Read train and test images and labels
    std::string train_image_directory = args.directory + "train/mnist_images";
    std::string train_label_directory =  args.directory + "train/mnist_labels.csv";
    std::string test_image_directory = args.directory + "test/mnist_images";
    std::string test_label_directory =  args.directory + "test/mnist_labels.csv";
    auto[train_h_images, srcWidth, srcHeight, numChannels, train_filenames] = read_images(train_image_directory);
    auto train_labels = readLabelsFromCSV(train_label_directory);
    auto[test_h_images, tsrcWidth, tsrcHeight, tnumChannels, test_filenames] = read_images(test_image_directory);
    auto test_labels = readLabelsFromCSV(test_label_directory);

    // Construct the augmentor
    ImageAugmentation Augmentor(srcWidth, srcHeight, args.dstWidth, args.dstHeight, numChannels);

    // Construct the network
    CNN CNNModel(cudnn, cublas, args.dstHeight, args.dstWidth,
                args.convfilterHeight, args.convfilterWidth,
                args.convstrideHeight, args.convstrideWidth,
                args.convpaddingHeight, args.convpaddingWidth,
                args.poolfilterHeight, args.poolfilterWidth,
                args.poolstrideHeight, args.poolstrideWidth,
                args.poolpaddingHeight, args.poolpaddingWidth,
                args.numFilters, numChannels,
                args.hiddenDim, args.numClass,
                args.batchSize, args.learningRate,
                args.weight_decay, args.dropoutProbability);

    // Train the model
    train(train_h_images, train_labels, test_h_images, test_labels, train_filenames, CNNModel, Augmentor, numChannels, args.dstHeight, args.dstWidth, args.epochs, args.batchSize, args.numClass, args.debug, args.save_directory);

    // Load model weights 
    CNNModel.LoadModelWeights(args.load_directory);

    // Test the model
    auto accuracy = test(test_h_images, test_labels, CNNModel, Augmentor, numChannels, args.dstHeight, args.dstWidth, args.batchSize, args.load_directory);

    // Cleanup
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);

    return 0;
}
