#include <stdio.h>
#include <cuda.h>

const int iNumberOfArrayElements = 5;

float *hostInput1 = NULL;
float *hostInput2 = NULL;  
float *hostOutput = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

// CUDA kernel
__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < len)
    {
        out[i] = in1[i] + in2[i];
    }
}

// entry point function
int main(void)
{
    void cleanup(void);

    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result = cudaSuccess;

    // host memory allocation
    hostInput1 = (float*)malloc(size);
    if(hostInput1 == NULL)
    {
        printf("Host Memory allocations is failes for hostInput1 array. \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if(hostInput2 == NULL)
    {
        printf("Host Memory allocations is failes for hostInput2 array. \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float*)malloc(size);
    if(hostOutput == NULL)
    {
        printf("Host Memory allocations is failes for hostOutput array. \n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into host arrays
    hostInput1[0] = 101.0;
    hostInput1[1] = 102.0;
    hostInput1[2] = 103.0;
    hostInput1[3] = 104.0;
    hostInput1[4] = 105.0;

    hostInput2[0] = 201.0;
    hostInput2[1] = 202.0;
    hostInput2[2] = 203.0;
    hostInput2[3] = 204.0;
    hostInput2[4] = 205.0;

    // device memory allocation
    result = cudaMalloc((void**)&deviceInput1, size);
    if(result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceInput2, size);
    if(result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceOutput, size);
    if(result != cudaSuccess)
    {
        printf("Device Memory allocation is failed for deviceOutput array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // copy data from host arrays into device arrays
    result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
        printf("Host to device data copy is failed for deviceInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
        printf("Host to device data copy is failed for deviceInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // kernel configuration
    dim3 dimGrid = dim3(1, 1, 1);
    dim3 dimBlock = dim3(iNumberOfArrayElements, 1, 1);

    // cuda kernal for vector addition
    vecAddGPU <<< dimGrid, dimBlock >>> (deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

    // copy data from device array into host array
    result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    if(result != cudaSuccess)
    {
        printf("device to host data copy is failed for hostoutput array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // vector addition on host
    for(int i = 0; i < iNumberOfArrayElements; i++)
    {
        printf(" %f + %f = %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
    }

    cleanup();
    return(0);
}

void cleanup(void)
{
    if(deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = NULL;
    }

    if(deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2 = NULL;
    }

    if(deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1 = NULL;
    }

    if(hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if(hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if(hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}


