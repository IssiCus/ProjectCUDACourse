/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "polarizationCUDA.h"

 // for load/save bmp files 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    std::string inputFolder = "Raw Images";
    std::string outputFolder = "Polarization Angle";

    // Create output folder if it doesn't exist
    if (CreateDirectory(outputFolder.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS) {
        std::cout << "Output folder is ready: " << outputFolder << std::endl;
    }
    else {
        std::cerr << "Failed to create output folder: " << outputFolder << std::endl;
        return 1;
    }

    // Iterate over all .bmp files in the input folder
    std::string searchPath = inputFolder + "\\*.bmp";
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "No .bmp files found in " << inputFolder << std::endl;
        return 1;
    }

    int fileCount = 0;
    do {
        if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

        ++fileCount;
        std::string fileName = findFileData.cFileName;
        std::string inputPath = inputFolder + "\\" + fileName;
        std::string outputPath = outputFolder + "\\" + fileName.substr(0, fileName.find_last_of(".")) + "_angle.bmp";

        std::cout << "Processing file #" << fileCount << ": " << inputPath << std::endl;

        try {
            processImages(inputPath, outputPath);
            std::cout << "Successfully processed: " << outputPath << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing file " << inputPath << ": " << e.what() << std::endl;
        }

    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);

    return 0;
}

void processImages(const std::string& inputPath, const std::string& outputPath)
{
    // Create buffers
    unsigned char* d_output = nullptr; 
    unsigned char* d_input = nullptr;  
    int width = 0, height = 0;

    try {

        // Load image to GPU
        loadImageToGPU(inputPath, &d_input, &width, &height);
        cudaMalloc(&d_output, width * sizeof(unsigned char)*height);

        // Launch kernel with a 8x8 thread block size
        dim3 blockSize(8, 8);
        
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);

        std::cout << "Grid Size: " << gridSize.x << " x " << gridSize.y << std::endl;

        polarizationAngle <<< gridSize, blockSize >>> (d_input, d_output, width, height);

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA Kernel execution failed.");
        }

        cudaDeviceSynchronize(); 

        // Save the processed image
        saveGPUimage(outputPath, d_output, width, height);
        std::cout << "Saved image: " << outputPath << std::endl;
        
        // Free GPU memory
        cudaFree(d_input);
        cudaFree(d_output);

    }
    catch (const std::exception& e)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << e.what() << std::endl;
        std::cerr << "Aborting." << std::endl;

        if (d_input) cudaFree(d_input);

        exit(EXIT_FAILURE);
    }


}


void loadImageToGPU(const std::string filePath, unsigned char** d_image, int* width, int* height) 
{
    // Load image using a BMP loader (stb_image)
    int channels;
    unsigned char* h_image = stbi_load(filePath.c_str(), width, height, &channels, 0);

    if (!h_image) {
        throw std::runtime_error("Failed to load image: " + filePath);
    }

    // Allocate memory for a single-channel grayscale image
    std::vector<unsigned char> grayscale_image((*width) * (*height));

    // Convert the image to grayscale 
    if (channels == 3 || channels == 4) { // RGB or RGBA
        for (int y = 0; y < *height; ++y) {
            for (int x = 0; x < *width; ++x) {
                int idx = y * (*width) * channels + x * channels;

                // Extract RGB values
                unsigned char r = h_image[idx];
                unsigned char g = h_image[idx + 1];
                unsigned char b = h_image[idx + 2];

                // Calculate grayscale using a weighted sum
                unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

                // Assign grayscale value to output image
                grayscale_image[y * (*width) + x] = gray;
            }
        }
    }
    else if (channels == 1) { // Already grayscale
        std::memcpy(grayscale_image.data(), h_image, (*width) * (*height));
    }
    else {
        stbi_image_free(h_image);
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }

    // Allocate memory on GPU
    cudaMalloc(reinterpret_cast<void**>(d_image), (*width)* (*height) * sizeof(unsigned char));

    // Copy image data to GPU
    cudaMemcpy(*d_image, grayscale_image.data(), (*width) * sizeof(char)*(*height), cudaMemcpyHostToDevice);

    // Free host memory
    stbi_image_free(h_image);
}


// Function to save the image from GPU  
void saveGPUimage(const std::string& filePath, unsigned char* d_image, int width, int height) {
    // Allocate host memory
    std::vector<unsigned char> h_image(width * height);

    // Copy image data from GPU to host
    cudaError_t err = cudaMemcpy(h_image.data(), d_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Use stb_image_write to save the image as BMP

    if (stbi_write_bmp(filePath.c_str(), width, height, 1, h_image.data())) {
        std::cout << "Image saved successfully: " << filePath << std::endl;
    }
    else {
        std::cerr << "Failed to save image: " << filePath << std::endl;
    }
}


// CUDA kernel to compute the polarization angle and conver it in grey scale
__global__ void polarizationAngle(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double p00 = 0, p90 = 0, p45 = 0, p135 = 0, s2 = 0, s1 = 0;

    if (x >= width || y >= height) return; // Ensure threads stay within bounds

    int idx = y * width + x;

    // Handle boundary pixels
    if (x == width - 1 || y == height - 1) {
        d_output[idx] = 0;
        return;
    }

    // Extract the intensity of different polarization in the macropixel 
    if (!(x % 2) && !(y % 2)) // top-left pixel => 90 degrees 
    {
        p90 = d_input[idx];
        p45 = d_input[idx + 1];
        p135 = d_input[idx + width];
        p00 = d_input[idx + width + 1];
        
    }

    else if ((x % 2) && !(y % 2)) // bottom-left pixel => 45 degrees
    {
        p45 = d_input[idx];
        p90 = d_input[idx + 1];
        p00 = d_input[idx + width];
        p135 = d_input[idx + width + 1];
      
    }
    else if ((x % 2) && (y % 2)) // bottom-right pixel => 0 degrees
    {
        p00 = d_input[idx];
        p135 = d_input[idx + 1];
        p45 = d_input[idx + width];
        p90 = d_input[idx + width + 1];
    
    }

    else if (!(x % 2) && (y % 2)) // // top-left pixel => 135 degrees
    {
        p135 = d_input[idx];
        p00 = d_input[idx + 1];
        p90 = d_input[idx + width];
        p45 = d_input[idx + width + 1];

    }
    
    s1 = p00 - p90;
    s2 = p45 - p135;

    /* __device__ double atan2(double y, double x) :  Result will be in radians, in the interval [-pi, +pi] */
    double angle_rad = 0.5 * atan2(s2, s1);

    // Normalize the angle to a 0-255 BW value
    d_output[idx] = static_cast<unsigned char>((angle_rad + 3.14/2) / 3.14 * 255.0);
    
}
