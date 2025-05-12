#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


int main() {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Shared Mem per block : %lu\n", devProp.sharedMemPerBlock);
    printf("Max threads per block : %d\n", devProp.maxThreadsPerBlock);
    printf("Max Grid Size : %d X %d X %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("Max num blocks : %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("Total Multi Processors : %d\n", devProp.multiProcessorCount);
}