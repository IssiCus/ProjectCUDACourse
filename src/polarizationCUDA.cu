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


/* standard CUDA's hardware limits: maximum blockDim 1024, gridDim 2to31 in x, 2to16 in y */
#include "polarizationCUDA.h"

 // for load/save bmp files 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[]) {

    bool computeAngle = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "true") computeAngle = true;
    }
    std::string inputFolder = "Raw Images";
    std::string outputFolder = "Output Images";

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
        std::string outputPath;
        std::string outputPathFFT;
        
        if(computeAngle)
            outputPath = outputFolder + "\\" + fileName.substr(0, fileName.find_last_of(".")) + "_angle.bmp";
        else
        {
            outputPath = outputFolder + "\\" + fileName.substr(0, fileName.find_last_of(".")) + "_intensity.bmp";
            outputPathFFT = outputFolder + "\\" + fileName.substr(0, fileName.find_last_of(".")) + "_FFT.bmp";
        }
        std::cout << "Processing file #" << fileCount << ": " << inputPath << std::endl;

        try {
            processImages(inputPath, outputPath, outputPathFFT, computeAngle);
            std::cout << "Successfully processed: " << outputPath << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing file " << inputPath << ": " << e.what() << std::endl;
        }

    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);

    return 0;
}

void processImages(const std::string& inputPath, const std::string& outputPath, const std::string& outputPathFFT, bool computeAngle)
{
    
    unsigned char* d_output = nullptr; 
    unsigned char* d_input = nullptr;  
    int width = 0, height = 0;


    try {
        

        // Load image to GPU
        loadImageToGPU(inputPath, &d_input, &width, &height);

        // Launch kernel with a 8x8 thread block size
        dim3 blockSize(8, 8);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);
        
        std::cout << "Grid Size: " << gridSize.x << " x " << gridSize.y << std::endl;
        if (computeAngle)
        {
           cudaMalloc(&d_output, width * sizeof(unsigned char) * height);
           polarizationAngle <<< gridSize, blockSize >>> (d_input, d_output, width, height);
           cudaDeviceSynchronize();

           // Save the processed image
           saveImage(outputPath, d_output, width, height);
           //std::cout << "Saved image: " << outputPath << std::endl;
        }
        else
        {
            // Compute intesity and crop the image to side x side where side = min(height, width)
            int side = std::min(width, height);
            int crop_x = (width - side) / 2;
            int crop_y = (height - side) / 2; 

            cudaMalloc(&d_output, side * sizeof(unsigned char) * side);


            float* d_intensity = nullptr;
            cudaMalloc(&d_intensity, side * sizeof(float) * side);
            intensityFromRaw <<< gridSize, blockSize >>> (d_input, d_output, d_intensity, width, height, side, crop_x, crop_y);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error in intensityFromRaw kernel: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("Kernel execution failed.");
            }

            //*** Compute FFT ***
            
            // Allocate memory
            const int FFT_OUT_WIDTH = side / 2 + 1; // Output width from R2C
            cufftComplex* d_complex;              
            cudaMalloc((void**)&d_complex, side * FFT_OUT_WIDTH * sizeof(cufftComplex)); // FFT output (complex data) device          


            dim3 gridSize((FFT_OUT_WIDTH + blockSize.x - 1) / blockSize.x,
                (side + blockSize.y - 1) / blockSize.y);

            // Create a 2D FFT plan

            cudaEvent_t start_event, stop_event;
            float time;
            cufftHandle plan_r2c;
            cufftPlan2d(&plan_r2c, side, side, CUFFT_R2C);

            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);

            cudaEventRecord(start_event, 0);

            // Execute forward R2C FFT
            cufftExecR2C(plan_r2c, d_intensity, d_complex);
            cudaDeviceSynchronize();

            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&time, start_event, stop_event);

            std::cout << "Enlapsed time: " << time << std::endl;

            // Clean up events Timing events
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error in cufftExecR2C: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("FFT execution failed.");
            }
                  
            // Compute the magnitude of the FFT to save the image

            // Allocate memory for magnitude in the device 
            float* d_magnitude;
            cudaMalloc((void**)&d_magnitude, side * FFT_OUT_WIDTH * sizeof(float));
           

            computeMagnitude << <gridSize, blockSize >> > (d_complex, d_magnitude, FFT_OUT_WIDTH);
            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Error in computeMagnitude kernel: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("Kernel execution failed.");
            }

            // Copy the magnitude matrix to host memory and save image
            saveFFT(outputPathFFT, d_magnitude, FFT_OUT_WIDTH*side, FFT_OUT_WIDTH, side);            
            
            // Save the processed image
            saveImage(outputPath, d_output, side, side);
            //std::cout << "Saved image: " << outputPath << std::endl;

            // Free GPU memory
            cudaFree(d_complex);
            cudaFree(d_intensity);

        }

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA Kernel execution failed.");
        }

               
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
        if (d_output) cudaFree(d_output);

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


// Functions to save the images from GPU  


void saveFFT(const std::string& filePath, float* d_image, int FFT_OUT_SIZE, int width, int height) {
    // Allocate host memory
    float* h_image = new float[FFT_OUT_SIZE];

    // Copy image data from GPU to host
    cudaError_t err = cudaMemcpy(h_image, d_image, FFT_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed in saveFFT: " << cudaGetErrorString(err) << std::endl;
        delete[] h_image;
        throw std::runtime_error("Failed to copy image data from device to host.");
    }

  
    // Normalize the image to unsigned char [0, 255]
    short* normalizedImage = new short[FFT_OUT_SIZE];
    float maxVal = 0.0f;

    for (int i = 0; i < FFT_OUT_SIZE; i++) {
        if(h_image[i] < 10000)
            if (h_image[i] > maxVal) maxVal = h_image[i];
    }

    for (int i = 0; i < FFT_OUT_SIZE; i++) {
        normalizedImage[i] = ((h_image[i] / maxVal) * 255.0f);
        /*if (!(i % 1000))
            std::cout << normalizedImage[i] << "vs" << h_image[i] << std::endl;*/
    }
    
    
    // Use stb_image_write to save the image as BMP

    if (stbi_write_bmp(filePath.c_str(), width, height, 1, normalizedImage)) {
       // std::cout << "Image saved successfully: " << filePath << std::endl;
    }
    else {
        std::cerr << "Failed to save image: " << filePath << std::endl;
    }

    // Free memory
    delete[] h_image;
}


void saveImage(const std::string& filePath, unsigned char* d_image, int width, int height) {
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
        //std::cout << "Image saved successfully: " << filePath << std::endl;
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

// CUDA kernel to compute the intensity and conver it in 8-bit grey scale
__global__ void intensityFromRaw(unsigned char* d_input, unsigned char* d_output, float * d_intensity, int width, int height, int side, int crop_x, int crop_y) 
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width-1 || y >= height-1) return; // Ensure threads stay within bounds of inputs


    if (x < side && y < side) // Ensure threads stay within bounds of outputs
    {
        int idx = (crop_y + y) * width + (crop_x + x);;
        int output_idx = y * side + x; 
        d_intensity[output_idx] = (d_input[idx] + d_input[idx + 1] + d_input[idx + width] + d_input[idx + 1 + width]) / 2;

        // Normalize the intensity to a 0-255 BW value:
        d_output[output_idx] = static_cast<unsigned char>(d_intensity[output_idx] / 511.0 * 255.0);
    }
    

}

__global__ void computeMagnitude(const cufftComplex* input, float* output, int FFT_OUT_WIDTH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int idx = x * FFT_OUT_WIDTH + y;
    if (x < FFT_OUT_WIDTH && y < ((FFT_OUT_WIDTH -1)*2)) {
        float real = input[idx].x; // Real part
        float imag = input[idx].y; // Imaginary part
        output[idx] = sqrtf(real * real + imag * imag); // Magnitude


    }
}


/*

The kernel intensityFromRaw and FFT calculations use thread blocks of size 8x8. Ensure that this choice provides enough threads per block for optimal GPU utilization. Use occupancy calculators or profiling tools to verify this​​.
For computeMagnitude, you hardcoded 256 threads per block. This is fine if it aligns with the target architecture's warp size and occupancy. However, ensure the block size doesn't exceed 1024 threads per block, as per CUDA's hardware limits​​.
Check clean/delete



Future improvements: The device memory allocation and deallocation (cudaMalloc and cudaFree) for images occur inside the processImages function. Repeated allocation and deallocation can impact performance. Consider allocating b. The FFT plan (cufftPlan2d) is created and destroyed for each image. If the image dimensions remain constant, create the plan once outside the loop and reuse it. Reusing plans can significantly reduce overhead​​​.
Use pinned memory for host-to-device transfers to improve data transfer speed between host and GPU​​.  cropToSquareCPU in GPU*/