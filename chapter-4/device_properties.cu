#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Device Count: %d\n", devCount);

    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
    }
    printf("Information\n");
    printf("Max Threads Per Block: %d\n", devProp.maxThreadsPerBlock);
    printf("Number of SMs: %d\n", devProp.multiProcessorCount);
    printf("Clock Rate: %d\n", devProp.clockRate);
    printf("Max threads in X-Dim: %d\n", devProp.maxThreadsDim[0]);
    printf("Max threads in Y-Dim: %d\n", devProp.maxThreadsDim[1]);
    printf("Max threads in Z-Dim: %d\n", devProp.maxThreadsDim[2]);
    printf("Registers per SM: %d\n", devProp.regsPerBlock);
    printf("Warp Size: %d\n", devProp.warpSize);

}