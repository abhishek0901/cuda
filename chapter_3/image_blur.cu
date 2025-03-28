#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <thread>
#include <chrono>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void blur_image(float *input, float *output, int rows, int cols, int blur_size) {
    int row_index = blockDim.y * blockIdx.y + threadIdx.y;
    int col_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_index < rows && col_index < cols) {
        float p_val = 0;
        float p_count = 0;
        for (int i = -blur_size; i < blur_size + 1; i++) {
            for (int j = -blur_size; j<blur_size + 1; j++) {
                int in_index_i = row_index + i;
                int in_index_j = col_index + j;
                if (in_index_i >= 0 && in_index_i < rows && in_index_j >= 0 && in_index_j < cols) {
                    int index = in_index_i * cols + in_index_j;
                    p_val += input[index];
                    p_count++;
                }
            }
        }

        output[row_index * cols + col_index] = p_val/p_count;
    }
}


void blur_image_cpu(float *input, float *output, int rows, int cols, int blur_size) {
    for (int i =0; i < rows; i++) {
        for (int j =0; j < cols; j++) {
            float p_val = 0;
            float p_count = 0;
            for (int k = -blur_size; k < blur_size + 1; k++) {
                for (int l = -blur_size; l < blur_size + 1; l++) {
                    int row_index = i + k;
                    int col_index = j + l;
                    if (row_index >= 0 && row_index < rows && col_index >= 0 && col_index < cols) {
                        p_val += input[row_index * cols + col_index];
                        p_count++;
                    }
                }
            }
            output[i * cols + j] = p_val/p_count;
        }
    }
}

int main() {
    int rows = 8192, cols = 8192, blur_size = 5;

    float *h_img, *h_blur_img, *d_img, *d_blur_img;

    //1. Allocate memory to host
    h_img = (float*)malloc(rows * cols * sizeof(float));
    h_blur_img = (float*)malloc(rows * cols * sizeof(float));

    //2. Init array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_img[i * cols + j] = (float)(std::rand()) / RAND_MAX;
            //h_img[i * cols + j] = (i * cols + j);
        }
    }

    //3. Create device memory
    cudaMalloc((void**)&d_img, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_blur_img, rows * cols * sizeof(float));

    //4. Copy host to device
    cudaMemcpy(d_img, h_img, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    //5. Run Kernel
    dim3 dimBlock(32,32,1);
    dim3 dimGrid(ceil(cols/32.0), ceil(rows/32.0), 1);
    blur_image<<<dimGrid, dimBlock>>>(d_img, d_blur_img, rows, cols, blur_size);

    //6. Check error from kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    //7. Synchronize all devices
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to GPU generate:  %3.1f ms \n", time);

    //8. copy output from device to host
    cudaMemcpy(h_blur_img, d_blur_img, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    //8.1 Run on CPU
    float *cpu_run;
    cpu_run = (float*)malloc(rows * cols * sizeof(float));
    gettimeofday(&t1, 0);
    blur_image_cpu(h_img, cpu_run, rows, cols, blur_size);
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    printf("Time to CPU generate:  %3.1f ms \n", time);

    //8.2 Correctness
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         int index = i * cols + j;
         if (std::abs(cpu_run[index] - h_blur_img[index]) > 1e-3) {
            printf("%d,%d Results mismatch\n", i,j);
            printf("CPU : %f, GPU : %f\n", cpu_run[index], h_blur_img[index]);
         }
      }
   }


    //9. Free Memory
    free(h_img);
    free(h_blur_img);
    free(cpu_run);
    cudaFree(d_img);
    cudaFree(d_blur_img);

    return 0;
}
