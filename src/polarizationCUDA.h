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
#include <sstream>
#include <math_functions.h>  // CUDA Math Library



void processImages(const std::string& inputPath, const std::string& outputPath);
void loadImageToGPU(const std::string filePath, unsigned char** d_image, int* width, int* height);
void saveGPUimage(const std::string& filePath, unsigned char* d_image, int width, int height);
__global__ void polarizationAngle(unsigned char* d_input, unsigned char* d_output, int width, int height);
#endif // POLARIZATIONCUDA_H