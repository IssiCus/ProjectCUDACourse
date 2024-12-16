# ProjectCUDAcourse
Angle of Polarization using CUDA

## Overview
This project demonstrates the use of CUDA to compute the polarization angle in images acquired with a polarization camera and saved in .bmp format. 
The goal is to utilize GPU acceleration to efficiently perform the computation. The project is a part of the CUDA at Scale for the Enterprise course. 
## Key Concepts
Performance Strategies, Image Processing, Polarization Camera
## Supported OSes
Windows
## Requirements 
- **Operating System**: Windows 10 or later 
- **Development Environment**: Visual Studio (2022) 
- **Dependencies**:  stb_image.h, stb_image_write.h  
- **Prerequisites**: Download and install the CUDA Toolkit 11.4 for your corresponding platform. 
## Installation 
1. Clone the repository: git clone https://github.com/IssiCus/ProjectCUDAcourse 
Open the solution file (.sln) in Visual Studio: polarizationCUDA.sln
2. Build the solution:
- Select the desired build configuration (Debug or Release).
- Press Ctrl+Shift+B to build the project.
3. Run the executable:
- After building, navigate to the bin or Debug/Release folder to find the compiled executable.
- Double-click the executable or run it from the command line:
path-to-your-executable/polarizationCUDA.exe 

## Usage
1.	Prepare the raw data:
o	Place raw image files from a polarization camera into the "Raw Images" folder.
2.	Run the program:
o	Execute the program as described in the Installation section.
3.	View the results:
o	The program processes the images and saves the results in the "Polarization Angle" folder.
o	If the folder does not exist, the program automatically creates it.

## More information:
You can find more information about the computation and comments about the provided example in “Code explanation.txt”

