#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
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

__global__ void color_to_rgb(float *color_image, float *gray_image, int rows, int cols) {
   int col_index = blockDim.x * blockIdx.x + threadIdx.x;
   int row_index = blockDim.y * blockIdx.y + threadIdx.y;

   if (row_index < rows && col_index < cols) {
      int index = row_index * cols + col_index;
      float R = color_image[index*3];
      float G = color_image[index*3 + 1];
      float B = color_image[index*3 + 2];
      gray_image[index] = 0.299 * R + 0.587 * G + 0.114 * B;
   }
}

void cpu_run(float *color_image, float *gray_image, int rows, int cols) {
   for (int i =0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         int index = i * cols + j;
         float R = color_image[index*3];
         float G = color_image[index*3 + 1];
         float B = color_image[index*3 + 2];
         gray_image[index] = 0.299 * R + 0.587 * G + 0.114 * B;
      }
   }
}



int main() {
   // Assume the image is 2D with R,G,B values
   int rows = 8192, cols = 8192;
   float *h_color_img, *d_color_img, *d_gray_img, *h_gray_img;
   
   // create host memory
   h_color_img = (float*)malloc(rows * cols * 3 * sizeof(float));
   h_gray_img = (float*)malloc(rows * cols * sizeof(float));

   // init array
   for (int i =0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
         h_color_img[i * cols + j] = (int)(std::rand() % 256);
      }
   }

   // create device memory
   cudaMalloc((void**)&d_color_img, rows * cols * 3 * sizeof(float));
   cudaMalloc((void**)&d_gray_img, rows * cols * sizeof(float));

   //copy memory from host to device
   cudaMemcpy(d_color_img, h_color_img, rows * cols * 3 * sizeof(float), cudaMemcpyHostToDevice);

   // call kernel
   dim3 dimGrid(ceil(cols/32.0), ceil(rows/32.0), 1);
   dim3 dimBlock(32,32,1);
   struct timeval t1, t2;

   gettimeofday(&t1, 0);

   color_to_rgb<<<dimGrid, dimBlock>>>(d_color_img, d_gray_img, rows, cols);
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
   }

   cudaDeviceSynchronize();
   gettimeofday(&t2, 0);
   double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

   printf("Time to GPU generate:  %3.1f ms \n", time);

   // copy memory from device to host
   cudaMemcpy(h_gray_img, d_gray_img, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

   printf("Task ran successfully!\n");

   // Test
   float *h_test_gray_img;
   h_test_gray_img = (float*)malloc(rows * cols * sizeof(float));
   gettimeofday(&t1, 0);
   cpu_run(h_color_img, h_test_gray_img, rows, cols);
   gettimeofday(&t2, 0);
   time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   printf("Time to CPU generate:  %3.1f ms \n", time);

   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         int index = i * cols + j;
         if (std::abs(h_test_gray_img[index] - h_gray_img[index]) > 1e-3) {
            printf("%d,%d Results mismatch\n", i,j);
            printf("CPU : %f, GPU : %f\n", h_test_gray_img[index], h_gray_img[index]);
         }
      }
   }

   //std::this_thread::sleep_for(std::chrono::seconds(10));


   // free host memory
   free(h_color_img); //1GB
   free(h_gray_img); //.3GB
   free(h_test_gray_img); //.3GB
   //free device memory
   cudaFree(d_color_img); //1GB
   cudaFree(d_gray_img); //.3GB

   return 0;
}