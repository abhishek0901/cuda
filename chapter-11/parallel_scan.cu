#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>

#define BLOCK_SIZE 6
#define COARSENING_FACTOR 4
#define GRID_SIZE 4


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
        A[i] = i;
    }
}

__global__  void coarsened_kogge_stone_prefix_sum(int *A, int size, int *B, int *S) {
    __shared__ int AB[COARSENING_FACTOR * BLOCK_SIZE];

    for (int cf = 0; cf < COARSENING_FACTOR; cf++) {
        int idx = blockIdx.x * (blockDim.x * COARSENING_FACTOR) + threadIdx.x + cf * blockDim.x;
        if (idx < size) {
            AB[threadIdx.x + cf * blockDim.x] = A[idx];
        } else {
            AB[threadIdx.x + cf * blockDim.x] = 0.0f;
        }
    }
    __syncthreads();

    // Start Phase 1
    int start_idx = threadIdx.x * COARSENING_FACTOR;
    int end_idx = start_idx + COARSENING_FACTOR;
    for (int idx = start_idx + 1; idx < end_idx; idx++) {
        AB[idx] += AB[idx - 1];
    }

    // Phase 2 : apply kogg stone on last elements
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (end_idx - 1 >= stride * COARSENING_FACTOR) {
            temp = AB[end_idx - 1] + AB[end_idx - 1 - stride * COARSENING_FACTOR];
        }
        __syncthreads();
        if (end_idx - 1 >= stride * COARSENING_FACTOR) {
            AB[end_idx - 1] = temp;
        }
    }

    __syncthreads();

    // Phase 3
    int last_end_idx = start_idx - 1;
    for (int idx = start_idx; idx < end_idx - 1; idx++) {
        if (last_end_idx >= 0) {
            AB[idx] += AB[last_end_idx];
        }
    }

    for (int idx = start_idx; idx < end_idx; idx++) {
        B[blockIdx.x * blockDim.x * COARSENING_FACTOR + idx] = AB[idx];
    }

    if (threadIdx.x == blockDim.x - 1) {
        S[blockIdx.x] = AB[COARSENING_FACTOR * BLOCK_SIZE -1];
    }

}

__global__ void merge_data(int *S, int *B, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(blockIdx.x > 0 && i < size){
        B[i] += S[blockIdx.x - 1];
    }
}


int main() {
    /*
        Assume you have an array of size S.
        You want to calculate the prefix sum of it.
        Block Dim = 1024 threads
        These Threads can process upto 1024 * 4 = 4096 element.
        

        Launch prefix sum kernel for each block.
        Record their results in global Y
        Call Prefix sum on this Y
        Sum the output on original X
    */

   int *A, *B, size = BLOCK_SIZE * COARSENING_FACTOR * GRID_SIZE;
   A = (int*)malloc(size * sizeof(int));
   B = (int*)malloc(size * sizeof(int));

   init_array(A, size);

   // GPU
   int *A_d, *B_d, *S_d;
   cudaMalloc((void**)&A_d, size * sizeof(int));
   cudaMalloc((void**)&B_d, size * sizeof(int));
   cudaMalloc((void**)&S_d, GRID_SIZE * sizeof(int));

   // Copy A
   cudaMemcpy(A_d, A, size * sizeof(int), cudaMemcpyHostToDevice);

   // Define Kernel
   dim3 dim_block(BLOCK_SIZE);
   dim3 dim_grid(GRID_SIZE);
   coarsened_kogge_stone_prefix_sum<<<dim_grid, dim_block>>>(A_d, size, B_d, S_d);
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
   }
   CUDA_CHECK(cudaDeviceSynchronize());

   int *S1_d;
   cudaMalloc((void**)&S1_d, sizeof(int));

   dim3 dim_block_1(GRID_SIZE);
   dim3 dim_grid_1(1);
   coarsened_kogge_stone_prefix_sum<<<dim_grid_1, dim_block_1>>>(S_d, GRID_SIZE, S_d, S1_d);
   cudaError_t err1 = cudaGetLastError();
   if (err1 != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
   }
   CUDA_CHECK(cudaDeviceSynchronize());

   dim3 dim_block_2(BLOCK_SIZE * COARSENING_FACTOR);
   dim3 dim_grid_2(GRID_SIZE);
   merge_data<<<dim_grid_2, dim_block_2>>>(S_d, B_d, size);
   cudaError_t err2 = cudaGetLastError();
   if (err2 != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
   }
   CUDA_CHECK(cudaDeviceSynchronize());


   cudaMemcpy(B, B_d, size * sizeof(int), cudaMemcpyDeviceToHost);

   for (int i =20; i < 20 + 8; i++) {
    printf("%d ", A[i]);
   }
   printf("\n");

   for (int i =20; i < 20 + 8; i++) {
    printf("%d ", B[i]);
   }
   printf("\n");


   free(A);
   free(B);
   cudaFree(A_d);
   cudaFree(B_d);
   cudaFree(S_d);
   cudaFree(S1_d);

}