#include "helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

// Error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel 1: Performs scan within each block
__global__ void scan_kernel(int *input, int *output, int *block_sums, int n) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    temp[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Kogge-Stone scan (inclusive)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write result
    if (idx < n)
        output[idx] = temp[tid];

    // Save block sum
    if (tid == blockDim.x - 1 && block_sums != nullptr)
        block_sums[blockIdx.x] = temp[tid];
}

// Kernel 2: Add scanned block sums to each element in block (except first block)
__global__ void add_block_sums(int *output, int *block_sums, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (blockIdx.x == 0 || idx >= n)
        return;

    output[idx] += block_sums[blockIdx.x - 1];
}

// Host function
void kogge_stone_scan(int *input_d, int *output_d, int n) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *block_sums_d = nullptr;
    int *scanned_block_sums_d = nullptr;

    if (num_blocks > 1) {
        CUDA_CHECK(cudaMalloc(&block_sums_d, num_blocks * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scanned_block_sums_d, num_blocks * sizeof(int)));
    }

    scan_kernel<<<num_blocks, BLOCK_SIZE>>>(input_d, output_d, block_sums_d, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Recursively scan block sums if necessary
    if (num_blocks > 1) {
        kogge_stone_scan(block_sums_d, scanned_block_sums_d, num_blocks);
        add_block_sums<<<num_blocks, BLOCK_SIZE>>>(output_d, scanned_block_sums_d, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(block_sums_d);
        cudaFree(scanned_block_sums_d);
    }
}


__global__ void inc_to_exc_converter(int *input_d, int *output_d, int size, int keep_last_elem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_size = keep_last_elem ? size + 1 : size;
    if (i < actual_size) {
        output_d[i] = (i == 0) ? 0 : input_d[i - 1];
    }
}

namespace helpers {

void parallel_scan_gpu(int *input, int *output, int size) {
    kogge_stone_scan(input, output, size);
}

void parallel_scan_cpu(int *input, int *output, int size) {
    const int N = size;
    int *h_input = input;
    int *h_output = output;

    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    parallel_scan_gpu(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

void inclusive_to_exclusive_scan(int *input_d, int *output_d, int size, bool keep_last_elem) {
    dim3 dim_block(BLOCK_SIZE);
    int block_size = keep_last_elem ? ceil((size + 1) * 1.0 / BLOCK_SIZE): ceil(size * 1.0 / BLOCK_SIZE);
    dim3 dim_grid(block_size);
    inc_to_exc_converter<<<dim_grid, dim_block>>>(input_d, output_d, size, keep_last_elem);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace helpers