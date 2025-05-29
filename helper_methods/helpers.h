#ifndef HELPERS_H
#define HELPERS_H

namespace helpers {
    // Function declaration
    void parallel_scan_cpu(int *input, int *output, int size);
    void parallel_scan_gpu(int *input_d, int *output_d, int size);
    void inclusive_to_exclusive_scan(int *input_d, int *output_d, int size, bool keep_last_elem);
    __host__ __device__ void merge_seq(int *A, int *B, int *C, int size_A, int size_B);
    __device__ int co_rank(int k, int *A, int size_A, int *B, int size_B);
    __device__ void simple_merge_gpu(int *A, int *B, int *C, int size_A, int size_B);
}

#endif