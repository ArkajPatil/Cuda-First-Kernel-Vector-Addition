# Cuda-First-Kernel-Vector-Addition

This is my first CUDA program.
It performs vector addition on the GPU using a CUDA kernel.

## What this program does
- Allocates memory on the CPU (Host)
- Allocates memory on the GPU (Device)
- Copies input data from Host to Device
- Launches a CUDA kernel
- Adds two vectors in parallel on GPU
- Copies the result from GPU back to the Host(CPU)
- Prints the output and frees all allocated memory


## CUDA Kernel
Inside kernel Function:   vecAddGPU()

Each CUDA thread adds one element of the two input vectors.


## Grid and Block Configuration
Grid  : 1 Block
Block : 5 Threads

Since the vector has only 5 elements, one block with 5 threads is enough.


## Requirements for compilation
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- Microsoft Visual Studio
- NVCC Compiler


## Compilation Process

Open **Developer Command Prompt for Visual Studio** 

Command for compilation:
'nvcc VectorAddition.cu -o VectorAddition.exe'

By this command .exe is created.

** If you don't have CUDA toolkit and other require components, 
     simply run the executable instead of compiling the .cu source file.

  

## Run

Simply run it on terminal:    'VectorAddition.exe'

