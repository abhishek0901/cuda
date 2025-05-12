#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>

#define NUM_BINS 128
#define NUM_THREADS 1024
#define BLOCK_SIZE 768

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void init_array(int *input, int size) {
    for (int i =0; i < size; i++) {
        // if ((float) rand() / RAND_MAX < 0.5) {
        //     input[i] = size - i;
        // } else {
        //     input[i] = size - i + 1;
        // }
        float val = (float) rand() / RAND_MAX * (NUM_BINS - 1);
        input[i] = (int)val;
        // input[i] = (float) rand() / RAND_MAX * (NUM_BINS - 1);
    }
}

void hist_sort_cpu(int *input, int *output, int size) {
    int *hist_bin;
    hist_bin = (int*)malloc(NUM_BINS * sizeof(int));

    for (int i = 0; i < NUM_BINS; i++) {
        hist_bin[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        int idx = input[i];
        hist_bin[idx] += 1;
    }

    int current_idx = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        int count = hist_bin[i];
        for (int j = 0; j < count; j++) {
            output[current_idx] = i;
            current_idx += 1;
        }
    }

    free(hist_bin);
}

__global__ void histogram(int *input, int *histogram, int size) {
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0.0f;
    }
    __syncthreads();

    // Histogram
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int idx = input[i];
        atomicAdd(&histo_s[idx], 1);
    }

    __syncthreads();
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int bin_value = histo_s[bin];
        if (bin_value > 0) {
            atomicAdd(&histogram[bin], bin_value);
        }
    }
}

void print_output(int *input, int *output, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
}


int main() {
    /*
        1. Define Input and Output array
        2. Write a CPU sorting kernel
        3. Verify the sorting algorithm
        4. Write a GPU sorting kernel
        5. Verify the sorting algorithm
        6. Compare time complexity for CPU vs GPU for arrays of size 1e9
    */

   // Declarartion
   struct timeval t1, t2;
   int size = 2000000000;
   int *input_array_h, *output_array_h, *histogram_array_h;
   int *input_array_d, *histogram_array;

   // Memory allocation
   input_array_h = (int*)malloc(size * sizeof(int));
   output_array_h = (int*)malloc(size * sizeof(int));
   histogram_array_h = (int*)malloc(NUM_BINS * sizeof(int));

   // Initialization
   init_array(input_array_h, size);

   // Call cpu kernel
   gettimeofday(&t1, 0);
   hist_sort_cpu(input_array_h, output_array_h, size);
   gettimeofday(&t2, 0);

   // Print results
   // print_output(input_array_h, output_array_h, size);

   // Look at time
   double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   printf("CPU Time to generate:  %3.1f ms \n", time);

   // GPU

   // GPU memory allocation
   CUDA_CHECK(cudaMalloc((void**)&input_array_d, size * sizeof(int)));
   CUDA_CHECK(cudaMalloc((void**)&histogram_array, NUM_BINS * sizeof(int)));

   // Copy elements to global memory
   CUDA_CHECK(cudaMemcpy(input_array_d, input_array_h, size * sizeof(int), cudaMemcpyHostToDevice));

   // Call kernel
   gettimeofday(&t1, 0);
   dim3 dim_block(NUM_THREADS);
   dim3 dim_grid(BLOCK_SIZE);

   histogram<<<dim_grid, dim_block>>>(input_array_d, histogram_array, size);
   //6. Check error from kernel
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
   }
   CUDA_CHECK(cudaDeviceSynchronize());

   // 7. Create output array
   cudaMemcpy(histogram_array_h, histogram_array, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
   int current_idx = 0;
   for (int i = 0; i < NUM_BINS; i++) {
      int count = histogram_array_h[i];
      for (int j = 0; j < count; j++) {
         output_array_h[current_idx] = i;
         current_idx += 1;
      }
   }

   gettimeofday(&t2, 0);
   time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   printf("GPU Time to generate:  %3.1f ms \n", time);


   //print_output(input_array_h, output_array_h, size);

   


   // Free memory
   cudaFree(input_array_d);
   cudaFree(histogram_array);
   free(input_array_h);
   free(output_array_h);
}