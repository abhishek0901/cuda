#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>


#define SCOPE 5
#define INPUT_TILE_DIM 6
#define OUTPUT_TILE_DIM (INPUT_TILE_DIM-2)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void init_function(float *function, int X, int Y, int Z, float h) {
    for (int z = 0; z < Z; z ++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                if (
                    z >= 1 && z < Z-1 &&
                    y >= 1 && y < Y-1 &&
                    x >= 1 && x < X-1
                ) {
                    function[z * Y * X + y * X + x] =  (float)pow(((x-1)*h), 3) + pow(((y-1)*h), 3) + pow(((z-1)*h), 3);
                } else {
                    function[z * Y * X + y * X + x] = 0.0f;
                }
            }
        }
    }
}

__global__ void first_order_derivative_gpu(
    float *function, 
    float *f_derivative, 
    unsigned int X, 
    unsigned int Y, 
    unsigned int Z, 
    float h) {
        int iStart = blockIdx.z * OUTPUT_TILE_DIM;
        int j = blockIdx.y*OUTPUT_TILE_DIM + threadIdx.y - 1;
        int k = blockIdx.x*OUTPUT_TILE_DIM + threadIdx.x - 1;

        __shared__ float inPrev_s[INPUT_TILE_DIM][INPUT_TILE_DIM];
        __shared__ float inCurr_s[INPUT_TILE_DIM][INPUT_TILE_DIM];
        __shared__ float inNext_s[INPUT_TILE_DIM][INPUT_TILE_DIM];

        if (
            iStart - 1 >= 0 && iStart - 1 < Z &&
            j >= 0 && j < Y &&
            k >= 0 && k < X
        ) {
            inPrev_s[threadIdx.y][threadIdx.x] = function[(iStart - 1) * Y * X + j * X + k];
        } else {
            inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (
            iStart >= 0 && iStart < Z &&
            j >= 0 && j < Y &&
            k >= 0 && k < X
        ) {
            inCurr_s[threadIdx.y][threadIdx.x] = function[iStart * Y * X + j * X + k];
        } else {
            inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for (int i = iStart; i < iStart + OUTPUT_TILE_DIM; i++) {
            if (
                i + 1 >= 0 && i + 1 < Z &&
                j >= 0 && j < Y &&
                k >= 0 && k < X
            ) {
                inNext_s[threadIdx.y][threadIdx.x] = function[(i+1) * Y * X + j * X + k];
            } else {
                inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            if (
                i >= 1 && i < Z - 1 &&
                j >= 1 && j < Y - 1 &&
                k >= 1 && k < X - 1
            ) {
                if (
                    threadIdx.y >= 1 && threadIdx.y < INPUT_TILE_DIM - 1 &&
                    threadIdx.x >= 1 && threadIdx.x < INPUT_TILE_DIM -1
                ) {
                    f_derivative[i * Y * X + j * X + k] = (
                                                            0 * inCurr_s[threadIdx.y][threadIdx.x]
                                                            + 1 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                                                            - 1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                                                            + 1 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                                                            - 1 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                                                            + 1 * inNext_s[threadIdx.y][threadIdx.x]
                                                            - 1 * inPrev_s[threadIdx.y][threadIdx.x]
                                                        )/(2*h);
                }
            }
            __syncthreads();
            inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
            inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
        }
}


int main() {
    float *function_h, *f_derivative_h, *function_d, *f_derivative_d;
    float h = 0.1;
    int X = SCOPE/h, Y = SCOPE/h, Z = SCOPE/h;
    function_h = (float*)malloc((Z+2) * (Y+2) * (X+2) * sizeof(float));
    f_derivative_h = (float*)malloc((Z+2) * (Y+2) * (X+2) * sizeof(float));

    cudaMalloc((void**)&function_d, (Z+2) * (Y+2) * (X+2) * sizeof(float));
    cudaMalloc((void**)&f_derivative_d, (Z+2) * (Y+2) * (X+2)* sizeof(float));

    // Init function
    init_function(function_h, X+2, Y+2, Z+2, h);

    // GPU
    cudaMemcpy(function_d, function_h, (Z+2) * (Y+2) * (X+2) * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel config
    dim3 dim_block(INPUT_TILE_DIM, INPUT_TILE_DIM);
    dim3 dim_grid(ceil(X * 1.0/OUTPUT_TILE_DIM), ceil(Y * 1.0/OUTPUT_TILE_DIM), ceil(Z * 1.0/OUTPUT_TILE_DIM));
    first_order_derivative_gpu<<<dim_grid, dim_block>>>(function_d, f_derivative_d, X+2, Y+2, Z+2, h);

    //Check error from kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    gpuErrchk(cudaDeviceSynchronize());
    
    // copy mem to host
    cudaMemcpy(f_derivative_h, f_derivative_d, (Z+2) * (Y+2) * (X+2) * sizeof(float), cudaMemcpyDeviceToHost);

    //get function derivative values
    int x = ceil((3)/h) + 1, y = ceil((3)/h) + 1, z = ceil((2)/h) + 1;
    printf("F(%d,%d,%d) = %f\n", x, y, z, function_h[z * (X+2) * (Y+2) + y * (X+2) + x]);
    printf("F'(%d,%d,%d) = %f\n", x, y, z,f_derivative_h[z * (X+2) * (Y+2) + y * (X+2) + x]);

    // Free
    free(function_h);
    free(f_derivative_h);
    cudaFree(function_d);
    cudaFree(f_derivative_d);
}