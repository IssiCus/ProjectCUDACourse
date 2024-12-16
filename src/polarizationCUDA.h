#ifndef POLARIZATIONCUDA_H
#define POLARIZATIONCUDA_H



#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>
#include <sstream>
#include <math_functions.h>  // CUDA's Math Library


void processImages(const std::string& inputPath, const std::string& outputPath, const std::string& outputPathFFT, bool computeAngle);
void loadImageToGPU(const std::string filePath, unsigned char** d_image, int* width, int* height);
void saveImage(const std::string& filePath, unsigned char* d_image, int width, int height);
void saveFFT(const std::string& filePath, float* d_image, int FFT_OUT_SIZE, int width, int height);
__global__ void polarizationAngle(unsigned char* d_input, unsigned char* d_output, int width, int height);
__global__ void intensityFromRaw(unsigned char* d_input, unsigned char* d_output, float* d_intensity, int width, int height, int crop_x, int crop_y, int side);
__global__ void computeMagnitude(const cufftComplex* input, float* output, int size);

#endif // POLARIZATIONCUDA_H