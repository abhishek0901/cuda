#include "helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace helpers {
    __host__ __device__ void merge_seq(int *A, int *B, int *C, int size_A, int size_B) {
        int i = 0;
        int j = 0;
        int k = 0;
        while ((i < size_A) && (j < size_B)) {
            if (A[i] <= B[j]) {
                C[k++] = A[i++];
            } else {
                C[k++] = B[j++];
            }
        }

        if (i == size_A) {
            while(j < size_B) {
                C[k++] = B[j++];
            }
        } else {
            while (i < size_A) {
                C[k++] = A[i++];
            }
        }
    }


    __device__ int co_rank(int k, int *A, int size_A, int *B, int size_B) {
        int i = k < size_A ? k : size_A;
        int j = k - i;
        int i_low = 0 > (k - size_B) ? 0 : k - size_B;
        int j_low = 0 > (k - size_A) ? 0 : k - size_A;
        int delta = 0;
        bool active = true;

        while (active) {
            if (i > 0 && j < size_B && A[i-1] >= B[j]) {
                delta = (i - i_low + 1) >> 1;
                j_low = j;
                j = j + delta;
                i = i - delta;
            } else if (j > 0 && i < size_A && B[j-1] > A[i]) {
                delta = (j - j_low + 1) >> 1;
                i_low = i;
                i = i + delta;
                j = j - delta;
            } else {
                active = false;
            }
        }
        return i;
    }

    __device__ void simple_merge_gpu(int *A, int *B, int *C, int size_A, int size_B) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int elementPerThread = ceil((size_A + size_B) * 1.0 / (blockDim.x * gridDim.x));
        int k_curr = tid * elementPerThread;
        int k_next = min((tid+1) * elementPerThread, size_A + size_B);

        int i_curr = co_rank(k_curr, A, size_A, B, size_B);
        int i_next = co_rank(k_next, A, size_A, B, size_B);
        int j_curr = k_curr - i_curr;
        int j_next = k_next - i_next;

        merge_seq(&A[i_curr], &B[j_curr], &C[k_curr], i_next - i_curr, j_next - j_curr);
    }
}