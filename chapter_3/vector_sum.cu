#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void vector_add(float *A, float *B, float *C, int M) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < M) {
        C[thread_index] = A[thread_index] + B[thread_index];
    }
}


void init_vector(float *A, int M) {
    for (int i = 0; i < M ; i++) {
        A[i] = (float) rand() / RAND_MAX;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int M = 1000000;
    size_t size = M * sizeof(float);

    // allocate mem to host
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // init vectors
    init_vector(h_A, M);
    init_vector(h_B, M);

    // Create mem on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // call kernel
    vector_add<<<ceil(M/BLOCK_SIZE), BLOCK_SIZE>>>(d_A, d_B, d_C, M);

    // copy from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Addition Complete...\n");

    // destroy memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}