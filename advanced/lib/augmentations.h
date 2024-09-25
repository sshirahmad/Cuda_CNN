#include <npp.h>
#include <nppi.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

class ImageAugmentation {
public:

    // Constructor
    ImageAugmentation(int srcWidth, int srcHeight, int newWidth, int newHeight, int numChannels);

    // Destructor
    ~ImageAugmentation();

    // Method to augment an image (assumed to be in linear memory)
    float* augment(float* h_image);

private:

    int srcWidth, srcHeight, numChannels; 
    int newWidth, newHeight;

    float* deviceInput; 
    float* deviceOutput;
    float* h_nppOutput;
    float* h_chwImage;

    void convertHWCtoCHW(const float* srcImage, float* chwImage);
    void normalize();
    void resize();
    void reset();
    void AllocateMemory();
    void FreeMemory();


};
