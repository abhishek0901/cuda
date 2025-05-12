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

__global__ void init_memory(int* blockCounter, int* flags, int* scan_value, int size) {
    *blockCounter = 0;
    for (int i = 0; i < size; i++) {
        flags[i] = 0;
        scan_value[i] = 0;
    }
    flags[0] = 1;
}

__global__ void single_scan(int *A, int size, int *B, int* blockCounter, int* flags, int* scan_value) {
    __shared__ int AB[COARSENING_FACTOR * BLOCK_SIZE];
    __shared__ unsigned int bid_s;
    __shared__ int previous_sum; 

    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads(); // all threads wait until first thread of this block executes

    unsigned int bid = bid_s;

    for (int cf = 0; cf < COARSENING_FACTOR; cf++) {
        int idx = bid * (blockDim.x * COARSENING_FACTOR) + threadIdx.x + cf * blockDim.x;
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

    // single pass scan
    if (threadIdx.x == 0) {
        while(atomicAdd(&flags[bid], 0) == 0){};
        previous_sum = scan_value[bid];
        scan_value[bid + 1] = previous_sum + AB[COARSENING_FACTOR * BLOCK_SIZE -1];
        __threadfence();
        atomicAdd(&flags[bid+1], 1);
    }
    __syncthreads();

    for (int idx = start_idx; idx < end_idx; idx++) {
        B[bid * blockDim.x * COARSENING_FACTOR + idx] = AB[idx] + previous_sum;
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
   int *A_d, *B_d, *block_counter, *flags, *scan_value;
   cudaMalloc((void**)&A_d, size * sizeof(int));
   cudaMalloc((void**)&B_d, size * sizeof(int));
   cudaMalloc((void**)&block_counter, sizeof(int));
   cudaMalloc((void**)&flags, GRID_SIZE * sizeof(int));
   cudaMalloc((void**)&scan_value, GRID_SIZE * sizeof(int));

   // Copy A
   cudaMemcpy(A_d, A, size * sizeof(int), cudaMemcpyHostToDevice);

   init_memory<<<1,1>>>(block_counter, flags, scan_value, GRID_SIZE);
   cudaError_t err_0 = cudaGetLastError();
   if (err_0 != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err_0));
   }
   CUDA_CHECK(cudaDeviceSynchronize());

   dim3 dim_grid(GRID_SIZE);
   dim3 dim_block(BLOCK_SIZE);
   single_scan<<<dim_grid, dim_block>>>(A_d, size, B_d, block_counter, flags, scan_value);
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
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
}