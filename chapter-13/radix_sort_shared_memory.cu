#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include "../helper_methods/helpers.h"
#include <random>

#define BLOCK_SIZE 1024
#define RADIX_VALUE 3
#define BUCKET_SIZE 1 << RADIX_VALUE

// Error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void init_array(int *input, int size) {
    //int const_array[16] = {4,3,1,2,1,2,5,7,8,1,3,4,5,9,11,17};
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<> dist(50, 8); // Normal distribution
    for (int i = 0; i < size; i++) {
        int num = dist(gen);
        input[i] = num >= 0 ? num : i;
        //input[i] = const_array[i];
    }
}

__global__ void build_block_level_bucket(int *input, int *output, int size, int iter) {
    /*
        - Create count of bits based on radix value.
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = blockIdx.x;
    int key, bit;
    if (idx < size) {
        key = input[idx];
        bit = (key >> iter) & ((1 << RADIX_VALUE) - 1);
        int actual_idx = start_idx + bit * gridDim.x;
        atomicAdd(&output[actual_idx], 1);
    }
}

__global__ void radix_sort_kernel(
        int *input, 
        int *output,
        int *block_bucket,
        int *block_bucket_exclusive_scan, 
        int size, 
        int iter
    ) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int bits[BLOCK_SIZE];
    __shared__ int input_temp[BLOCK_SIZE];
    int key, bit, actual_dst;
    if (idx < size) {
        input_temp[tid] = input[idx];
    }
    __syncthreads();
    for (int radix_val = 0; radix_val < RADIX_VALUE; radix_val++) {
        if (idx < size) {
            key = input_temp[tid];
            bit = ((key >> iter)>>radix_val) & 1;
            bits[tid] = bit;
        }

        // Iclusive Scan
        __syncthreads();
        // Kogge-Stone scan (inclusive)
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            int val = 0;
            if (tid >= offset)
                val = bits[tid - offset];
            __syncthreads();
            bits[tid] += val;
            __syncthreads();
        }

        if (idx < size) {
            int num_ones_before = tid > 0?bits[tid-1]:0;
            int N = blockIdx.x != gridDim.x - 1 ? blockDim.x : size - blockIdx.x * blockDim.x;
            int num_ones_total = bits[N - 1];
            int dst = (bit == 0)?(tid - num_ones_before):(N - num_ones_total + num_ones_before);
            input_temp[dst] = key;
        }
        __syncthreads();
    }
    // Put them in correct dst in global memory

    // Perform exclusive scan on bucket level
    /*
        - Since the number of buckets would be ~8-16. This can be done by one thread.
    */
    __shared__ int local_scan[BUCKET_SIZE];
    if (threadIdx.x == 0) {
        local_scan[0] = 0;
        for (int bucket = 1; bucket < BUCKET_SIZE; bucket++) {
            local_scan[bucket] = local_scan[bucket - 1] + block_bucket[(bucket-1) * gridDim.x + blockIdx.x];
        }
    }
    __syncthreads();
    if (idx < size) {
        key = input_temp[threadIdx.x];
        bit = (key >> iter) & ((1 << RADIX_VALUE) - 1);
        actual_dst = block_bucket_exclusive_scan[bit * gridDim.x + blockIdx.x] + threadIdx.x - local_scan[bit];
        output[actual_dst] = key;
    }

}

void radix_sort_gpu(int *input, int *output, int size) {
    int max_val;
    for (int i = 0; i < size; i++) {
        if (max_val < input[i])max_val = input[i];
    }
    int NUM_BLOCKS = ceil(size * 1.0/BLOCK_SIZE);
    int BUCKETS = BUCKET_SIZE;
    int *input_d, *output_d;
    cudaMalloc((void**)&input_d, size * sizeof(int));
    cudaMalloc((void**)&output_d, size * sizeof(int));
    cudaMemcpy(input_d, input, size * sizeof(int), cudaMemcpyHostToDevice);

    int *block_level_bucket_d, *block_level_bucket_exclusive_scan_d;
    cudaMalloc((void**)&block_level_bucket_d, BUCKETS * NUM_BLOCKS * sizeof(int));
    cudaMalloc((void**)&block_level_bucket_exclusive_scan_d, BUCKETS * NUM_BLOCKS * sizeof(int));
    int iter = 0, *current_ref = input_d;;

    while (max_val != 0) {
        // Block Level bucket sizes
        cudaMemset(block_level_bucket_d, 0, BUCKETS * NUM_BLOCKS * sizeof(int));
        build_block_level_bucket<<<NUM_BLOCKS, BLOCK_SIZE>>>(current_ref, block_level_bucket_d, size, iter);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate Global Position Table
        cudaMemset(block_level_bucket_exclusive_scan_d, 0, BUCKETS * NUM_BLOCKS * sizeof(int));
        helpers::parallel_scan_gpu(block_level_bucket_d, block_level_bucket_exclusive_scan_d, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        int *block_level_bucket_exclusive_scan_d_temp;
        cudaMalloc((void**)&block_level_bucket_exclusive_scan_d_temp, BUCKETS * NUM_BLOCKS * sizeof(int));
        helpers::inclusive_to_exclusive_scan(block_level_bucket_exclusive_scan_d, block_level_bucket_exclusive_scan_d_temp, size, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(block_level_bucket_exclusive_scan_d);
        block_level_bucket_exclusive_scan_d = block_level_bucket_exclusive_scan_d_temp;

        radix_sort_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            current_ref, 
            output_d, 
            block_level_bucket_d, 
            block_level_bucket_exclusive_scan_d, 
            size, 
            iter
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        current_ref = output_d;
        iter++;
        max_val = (max_val >> 1);
    }
    cudaMemcpy(output, output_d, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(input_d);
    cudaFree(output_d);
}


int main() {
    // Define Input array
    int *input, *output, size = 5000;
    input = (int*)malloc(size * sizeof(int));
    output = (int*)malloc(size * sizeof(int));

    // init array
    init_array(input, size);
    
    // call gpu sort
    radix_sort_gpu(input, output, size);

    for (int i = 4000; i < 4000+20; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");


    free(input);
    free(output);
}