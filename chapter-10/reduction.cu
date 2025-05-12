#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void max_reduction_cpu(int *input, int size, int *output) {
    int max = 0;
    for (int i = 0; i < size; i++) {
        if (max < input[i]) {
            max = input[i];
        }
    }
    *output  = max;
}


__global__ void coarsened_max_reduction_kernel(int *input, int *output, int size) {
    __shared__ int input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    int max_val = 0;
    if (i < size) {
        max_val = input[i];
    }

    for (unsigned int tile = 1; tile < 2 * COARSE_FACTOR; tile ++) {
        if (i + tile * BLOCK_DIM < size) {
            max_val = max(max_val, input[i + tile * BLOCK_DIM]);
        }
    }

    input_s[t] = max_val;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] = max(input_s[t], input_s[t + stride]);
        }
    }

    if (t == 0) {
        atomicMax(output, input_s[0]);
    }
}


void init_array(int *input, int size) {
    for (int i = 0; i < size; i++) {
        float val = (float) rand() / RAND_MAX * 10000;
        input[i] = (int)val;
        //input[i] = i;
    }
}

int main() {
    int *input_h, output = 0;
    int *input_d, *output_d;
    int size = 1000000;

    input_h = (int*)malloc(size * sizeof(int));
    init_array(input_h, size);


    cudaMalloc((void**)&input_d, size * sizeof(int));
    cudaMalloc((void**)&output_d, sizeof(int));
    cudaMemcpy(input_d,input_h, size * sizeof(int), cudaMemcpyHostToDevice);

    // max_reduction_cpu(input_h, size, &output);
    // printf("Max value : %d\n", output);

    dim3 dim_block(BLOCK_DIM);
    dim3 dim_grid(ceil(size * 1.0/(BLOCK_DIM * 2 * COARSE_FACTOR)));
    coarsened_max_reduction_kernel<<<dim_grid, dim_block>>>(input_d, output_d, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(&output, output_d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Max value : %d\n", output);


    free(input_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}