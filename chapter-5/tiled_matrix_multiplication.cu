#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define TILE_WIDTH 16
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
           // A[i*C+j] = (i+j);
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

__global__ void tiled_matrix_multiplication_gpu(float *A, float *B, float *C, int N, int K, int M) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    float p_val = 0;
    for (int ph = 0; ph < ceil(K/(float)TILE_WIDTH); ph++) {
        //Load part of data into shared memory
        if ((row < N) && (ph * TILE_WIDTH + tx) < K)
            Ads[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        else Ads[ty][tx] =0.0f;
        if (((ty + ph * TILE_WIDTH) < K) && col < M)
            Bds[ty][tx] = B[(ty + ph * TILE_WIDTH) * M + col];
        else Bds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            p_val += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < M)
        C[row*M+col] = p_val;
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
    int N = 100;
    int K = 50;
    int M = 150;

    // Allocate memory
    h_A = (float*)malloc(N * K * sizeof(float));
    h_B = (float*)malloc(K * M * sizeof(float));
    h_C_cpu = (float*)malloc(N * M * sizeof(float));
    h_C_gpu = (float*)malloc(N * M * sizeof(float));

    // Init array
    init_array(h_A, N , K);
    init_array(h_B, K , M);

    //CPU matrix multiplication
    //matrix_multi_cpu(h_A, h_B, h_C_cpu, N, K, M);

    // #### GPU #####
    float *d_A, *d_B, *d_C;

    // Alloc memory
    cudaMalloc((void**)&d_A, N * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * M * sizeof(float));
    cudaMalloc((void**)&d_C, N * M * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dim_block(TILE_WIDTH,TILE_WIDTH);
    dim3 dim_grid(ceil(M * 1.0f/TILE_WIDTH), ceil(N * 1.0f/TILE_WIDTH));
    gettimeofday(&t1, 0);
    tiled_matrix_multiplication_gpu<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N, K, M);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("GPU Time to generate:  %3.1f ms \n", time);

    // Copy to host
    cudaMemcpy(h_C_gpu, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

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