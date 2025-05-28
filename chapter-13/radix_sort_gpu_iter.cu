#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include "../helper_methods/helpers.h"

#define SIZE 40000
#define BLOCK_DIM 1024

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void init_array(int *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = size - i;
    }
}

bool is_sorted(int *A, int size) {
    bool cond = true;
    for (int i = 1; i < size; i++) {
        if (A[i] < A[i-1]){
            cond = false;
            break;
        }
    }
    return cond;
}

__global__ void radix_sort_stage1(int *A, int *B, int *bits, int size, int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bit;
    if (i < size){
        bit = (A[i] >> iter) & 1;
        bits[i] = bit;
    }
}

__global__ void radix_sort_stage3(int *A, int *B, int *bits, int size, int iter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int num_ones_before = bits[i];
        int num_ones_total = bits[size];
        int bit = (A[i] >> iter) & 1;
        int dst = (bit == 0)?(i - num_ones_before):(size - num_ones_total + num_ones_before);
        B[dst] = A[i];
    }
}

bool test_bit_calc(int *A_d, int *bits_d, int size, int iter) {
    bool res = true;
    int *A, *bits;
    A = (int*)malloc(size * sizeof(int));
    bits = (int*)malloc(size * sizeof(int));
    cudaMemcpy(A, A_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bits, bits_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        int expected_bit = (A[i] >> iter) & 1;
        if (bits[i] != expected_bit) {
            res = false;
            break;
        }
    }
    return res;
}

bool test_inclusive_scan(int *bits_d, int *bits_output_d, int size) {
    bool res = true;
    int current_val = 0, *bits_output, *bits;
    bits = (int*)malloc(size * sizeof(int));
    bits_output = (int*)malloc(size * sizeof(int));
    cudaMemcpy(bits, bits_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bits_output, bits_output_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        current_val +=  bits[i];
        if (current_val != bits_output[i]) {
            res = false;
            break;
        }
    }
    return res;
}

void radix_sort_gpu(int *A, int size, int *B) {
    int *A_d, *B_d, *bits_d, *bits_output_d, *bits_output_d_temp, max_val = 0;
    for (int i = 0; i < size; i++) {
        if (max_val < A[i])max_val = A[i];
    }
    //A_tmp = (int*)malloc(8 * sizeof(int));

    cudaMalloc((void**)&A_d, size * sizeof(int));
    cudaMalloc((void**)&B_d, size * sizeof(int));
    cudaMalloc((void**)&bits_d, size * sizeof(int));
    cudaMalloc((void**)&bits_output_d, size * sizeof(int));
    cudaMalloc((void**)&bits_output_d_temp, (size+1) * sizeof(int));

    cudaMemcpy(A_d, A, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_DIM);
    dim3 dim_grid(ceil(size * 1.0 / BLOCK_DIM));

    int iter = 0;
    int *current_ref = A_d;
    cudaError_t err;
    while (max_val != 0) {
        radix_sort_stage1<<<dim_grid, dim_block>>>(current_ref, B_d, bits_d, size, iter);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        helpers::parallel_scan_gpu(bits_d, bits_output_d, size);
        helpers::inclusive_to_exclusive_scan(bits_output_d, bits_output_d_temp, size);

        radix_sort_stage3<<<dim_grid, dim_block>>>(current_ref, B_d, bits_output_d_temp, size, iter);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        current_ref = B_d;
        iter++;
        max_val = (max_val >> 1);
    }

    cudaMemcpy(B, B_d, size * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(bits_d);
    cudaFree(bits_output_d);
}


int main() {
    int *A, size = SIZE, *B;
    A = (int*)malloc(size * sizeof(int));
    B = (int*)malloc(size * sizeof(int));
    init_array(A, size);
    radix_sort_gpu(A, size, B);
    printf("is_sorted : %d\n", is_sorted(B, size));
    for (int i = 0; i < 8; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");
    free(A);
    free(B);
}