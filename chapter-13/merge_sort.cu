#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include <random>
#include "../helper_methods/helpers.h"

#define MAX_THREADS_PER_BLOCK 2

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


__global__ void merge_sort(int *input, int *output, int size, int segment_size) {
    int tid = threadIdx.x;
    int start_first = 0, end_first = 0, start_second = 0, end_second = 0, start_third = 0;
    start_first = min(blockIdx.x * 2 * segment_size, size);
    start_second = min(start_first + segment_size, size);
    end_first = start_second;
    end_second = min(start_second + segment_size, size);
    start_third = start_first;

    // Actual Merging of elements
    int elementPerThread = ceil((end_first - start_first + end_second - start_second) * 1.0 / blockDim.x);
    int k_curr = tid * elementPerThread;
    int k_next = min((tid+1) * elementPerThread, end_first - start_first + end_second - start_second);
    int i_curr = helpers::co_rank(k_curr, &input[start_first], end_first - start_first, &input[start_second], end_second - start_second);
    int i_next = helpers::co_rank(k_next, &input[start_first], end_first - start_first, &input[start_second], end_second - start_second);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    helpers::merge_seq(&input[start_first + i_curr], &input[start_second + j_curr], &output[start_third + k_curr], i_next - i_curr, j_next - j_curr);
}


// Merge sort
void merge_sort_gpu(int *input, int *output, int size) {
    /*
        Start from SegmentSize S:= 1 -> size
        For each segment size S, launch merge_sort kernel which internally will merge arrays
    */
    int *input_d, *output_d, *output_d_1;
    cudaMalloc((void**)&input_d, size * sizeof(int));
    cudaMalloc((void**)&output_d, size * sizeof(int));
    cudaMalloc((void**)&output_d_1, size * sizeof(int));
    cudaMemcpy(input_d, input, size * sizeof(int), cudaMemcpyHostToDevice);
    int *current_ref = input_d;
    for (int segment_size = 1; segment_size < size; segment_size *= 2) {
        int dim_grid = ceil(size * 1.0 / (2 * segment_size));
        int dim_block = min(2 * segment_size, MAX_THREADS_PER_BLOCK); // Need to revisit
        merge_sort<<<dim_grid, dim_block>>>(current_ref, output_d, size, segment_size);
        current_ref = output_d;
        output_d = output_d_1;
        output_d_1 = current_ref;
    }
    cudaMemcpy(output, current_ref, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
}


int main() {
    int *input, *output, size = 16;
    input = (int*)malloc(size * sizeof(int));
    output = (int*)malloc(size * sizeof(int));

    // init array
    init_array(input, size);

    // call gpu sort
    merge_sort_gpu(input, output, size);

    for (int i = 0; i < 16; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");


    free(input);
    free(output);
}