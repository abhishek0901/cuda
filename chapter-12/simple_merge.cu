#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define NUM_THREADS 1024

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void init(int *A, int size, int buffer) {
    for (int i = 0; i < size; i++) {
        A[i] = i + buffer;
    }
}

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

__global__ void simple_merge_gpu(int *A, int *B, int *C, int size_A, int size_B) {
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

void print_info(int *A, int *B, int *C, int size_A, int size_B) {
    printf("########### A ################\n");
    for (int i = 0; i < size_A; i++) {
        printf("%d ", A[i]);
    }
    printf("\n########### B ################\n");
    for (int i = 0; i < size_B; i++) {
        printf("%d ", B[i]);
    }
    printf("\n########### C ################\n");
    for (int i = 0; i < size_A + size_B; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
}


int main() {
    int *A, *B, *C, size_A = 5000000, size_B=3000000;
    A = (int*)malloc(size_A * sizeof(int));
    B = (int*)malloc(size_B * sizeof(int));
    C = (int*)malloc((size_A + size_B) * sizeof(int));

    init(A, size_A, 3);
    init(B, size_B, 2);

    // merge_seq(A,B,C, size_A, size_B);
    //print_info(A,B,C, size_A, size_B);

    // GPU
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size_A * sizeof(int));
    cudaMalloc((void**)&B_d, size_B * sizeof(int));
    cudaMalloc((void**)&C_d, (size_A + size_B) * sizeof(int));

    // Memcpy
    cudaMemcpy(A_d, A, size_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_B * sizeof(int), cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dim_block(NUM_THREADS);
    dim3 dim_grid(ceil((size_A + size_B) * 1.0/ (NUM_THREADS * 32)));
    simple_merge_gpu<<<dim_grid, dim_block>>>(A_d, B_d, C_d, size_A, size_B);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(C, C_d, (size_A + size_B) * sizeof(int), cudaMemcpyDeviceToHost);

    print_info(A,B,C, 5, 5);


    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}