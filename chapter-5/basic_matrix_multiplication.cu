#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void init_array(float *A, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            A[i*C+j] = (float) rand() / RAND_MAX;
            //A[i*C+j] = (i+j);
        }
    }
}

void matrix_multi_cpu(float *A, float *B, float *C, int N, int K, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float pval = 0;
            for (int k = 0; k < K; k++) {
                pval += A[i*K+k] * B[k*M + j];
            }
            C[i*M + j] = pval;
        }
    }
}

void print_matrix_results(float *A, float *B, float *C, int N, int K, int M) {
    printf("########################################\n");
    printf("Matrix A\n");
    for (int i = 0; i < N; i++) {
        for (int j =0; j < K; j++) {
            printf("%f ", A[i*K+j]);
        }
        printf("\n");
    }
    printf("########################################\n");
    printf("Matrix B\n");
    for (int i = 0; i < K; i++) {
        for (int j =0; j < M; j++) {
            printf("%f ", B[i*M+j]);
        }
        printf("\n");
    }
    printf("########################################\n");
    printf("Matrix C\n");
    for (int i = 0; i < N; i++) {
        for (int j =0; j < M; j++) {
            printf("%f ", C[i*M+j]);
        }
        printf("\n");
    }
}

__global__ void memory_multiplication(float *A, float *B, float *C, int N, int K, int M) {
    int row_index = blockDim.y * blockIdx.y + threadIdx.y;
    int col_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_index < N && col_index < M) {
        float pval = 0;
        for (int k = 0; k < K; k++) {
            pval += A[row_index * K + k] * B[k * M + col_index];
        }
        C[row_index * M + col_index] = pval;
    }
}

bool validate_arrays(float *A, float *B, int N, int M) {
    bool is_matched = true;
    for (int i =0; i < N; i++) {
        for (int j =0; j < M; j++) {
            if (abs(A[i*M+j] - B[i*M+j]) > 1e-3) {
                printf("Array Mismatch at (%d, %d) - A: %f, B - %f\n", i, j, A[i*M+j], B[i*M+j]);
                is_matched = false;
            }
        }
    }
    return is_matched;
}


int main() {
    //time
    struct timeval t1, t2;

    // Define Host Arrays
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;

    // Define Array size
    int N = 10000;
    int K = 5000;
    int M = 15000;

    // Allocate memory
    h_A = (float*)malloc(N * K * sizeof(float));
    h_B = (float*)malloc(K * M * sizeof(float));
    h_C_cpu = (float*)malloc(N * M * sizeof(float));
    h_C_gpu = (float*)malloc(N * M * sizeof(float));

    // Init array
    init_array(h_A, N , K);
    init_array(h_B, K , M);

    //CPU matrix multiplication
    gettimeofday(&t1, 0);
    //matrix_multi_cpu(h_A, h_B, h_C_cpu, N, K, M);
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("CPU Time to generate:  %3.1f ms \n", time);

    //print results
    // print_matrix_results(h_A, h_B, h_C_cpu, N, K, M);

    //GPU
    // create device arrays
    float *d_A, *d_B, *d_C;

    // Allocate memory to device
    cudaMalloc((void**)&d_A, N * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * M * sizeof(float));
    cudaMalloc((void**)&d_C, N * M * sizeof(float));

    // Copy array to device
    cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    //Call kernel
    dim3 dim_block(16,16,1);
    dim3 dim_grid(ceil(M/16.0), ceil(N/16.0), 1);
    gettimeofday(&t1, 0);
    memory_multiplication<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N, K, M);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("GPU Time to generate:  %3.1f ms \n", time);
    //Copy meory from device to host
    cudaMemcpy(h_C_gpu, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    //Verify results
    //print_matrix_results(h_A, h_B, h_C_gpu, N, K, M);

    // if (validate_arrays(h_C_gpu, h_C_cpu, N, M)) {
    //     printf("They Matched\n");
    // }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}