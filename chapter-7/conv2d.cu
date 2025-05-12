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

#define FILTER_RADIUS 1
#define IN_TILE_DIM 16
#define CHANNEL 3
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

// Define constant Memory
__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

void init_array(float *input, int R, int C, int Z) {
   for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
         for (int z = 0; z < Z; z++) {
            input[z * R * C + r * C + c] = (float) rand() / RAND_MAX * 255;
            //input[z * R * C + r * C + c] = 1;
         }
      }
   }
}

void init_filter(float *input, int R, int C) {
   for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
         input[r * C + c] = -1;
      }
   }
   int center_r = (int)R/2, center_c = (int)C/2;
   input[center_r * C + center_c] = 8;
}

void print_array(float *A, int R, int C, int Z) {
   printf("########################################\n");
   for (int z = 0; z < Z; z++) {
      for (int r = 0; r < R; r++) {
         for (int c = 0; c < C; c++) {
            //printf("%f ", A[r * C * Z + c * Z + z]);
            printf("%f ", A[z * R * C + r * C + c]);
         }
         printf("\n");
      }
      printf("\n\n");
   }
}

void conv2d_cpu(float *input, float *filter, float *output, int R, int C) {
   for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
         int elem = 0;
         for (int z = 0; z < CHANNEL; z++) {
            for (int filter_r = -FILTER_RADIUS; filter_r < FILTER_RADIUS + 1; filter_r++) {
               for (int filter_c = -FILTER_RADIUS; filter_c < FILTER_RADIUS + 1; filter_c++) {
                  int input_row = r + filter_r, input_col = c + filter_c;
                  int filter_elem = filter[(filter_r + FILTER_RADIUS) * (2*FILTER_RADIUS+1) + (filter_c + FILTER_RADIUS)];
                  if (input_row >= 0 && input_row < R && input_col >= 0 && input_col < C) {
                     int input_elem = input[z * R * C + input_row * C + input_col];
                     elem += filter_elem * input_elem;
                  }
               }
            }
         }
         output[r * C + c] = elem/CHANNEL;
      }
   }
}

__global__ void conv2d_gpu(float *input, float *output, int R, int C) {
   int plane = blockIdx.z * OUT_TILE_DIM + threadIdx.z;
   int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
   int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

   // Loading input tile
   __shared__ float input_s[CHANNEL][IN_TILE_DIM][IN_TILE_DIM];
   if (
      row >= 0 && row < R &&
      col >=0 && col < C &&
      plane >=0 && plane < CHANNEL
   ) {
      input_s[threadIdx.z][threadIdx.y][threadIdx.x] = input[plane * R * C + row * C + col];
   } else {
      input_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
   }

   __syncthreads();

   //calculate output elements
   int tileCol = threadIdx.x - FILTER_RADIUS;
   int tileRow = threadIdx.y - FILTER_RADIUS;

   if (
      row >= 0 && row < R &&
      col >=0 && col < C &&
      threadIdx.z == 0
   ) {
      if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
         float Pvalue = 0.0f;
         for (plane = 0; plane < CHANNEL; plane++) {
            for (int filter_r = -FILTER_RADIUS; filter_r < FILTER_RADIUS + 1; filter_r++) {
               for (int filter_c = -FILTER_RADIUS; filter_c < FILTER_RADIUS + 1; filter_c++) {
                  int in_row = tileRow + filter_r + FILTER_RADIUS, in_col = tileCol + filter_c + FILTER_RADIUS;
                  if (in_row >= 0 && in_row < IN_TILE_DIM && in_col >=0 && in_col < IN_TILE_DIM) {
                     Pvalue += F[filter_r + FILTER_RADIUS][filter_c + FILTER_RADIUS] * 
                     input_s[plane][in_row][in_col];
                  }
               }
            }
         }
         output[row*C+col] = Pvalue / CHANNEL;
      }
   }
}

int main() {
   //time
    struct timeval t1, t2;
   // define vars
   float *input_array_h, *filter_h, *output_array_h, *input_array_d, *output_array_d, *output_array_h_gpu;

   //allocate memory -> A X B X C
   int R = 8192, C = 8192, Z= CHANNEL;
   input_array_h = (float*)malloc(R * C * Z * sizeof(float));
   output_array_h = (float*)malloc(R * C * sizeof(float));
   output_array_h_gpu = (float*)malloc(R * C * sizeof(float));
   cudaMalloc((void**)&input_array_d, R * C * Z * sizeof(float));
   cudaMalloc((void**)&output_array_d, R * C * Z * sizeof(float));

   // define filter_h -> 3 X 3
   filter_h = (float*)malloc((FILTER_RADIUS*2 + 1) * (FILTER_RADIUS*2 + 1)* sizeof(float));

   // init array
   init_array(input_array_h, R, C, Z);

   // init filter
   init_filter(filter_h, (FILTER_RADIUS*2 + 1), (FILTER_RADIUS*2 + 1));

   // cpu conv2d
   gettimeofday(&t1, 0);
   conv2d_cpu(input_array_h, filter_h, output_array_h, R, C);
   gettimeofday(&t2, 0);
   double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   printf("CPU Time to generate:  %3.1f ms \n", time);

   //gpu
   // copy memory to device
   cudaMemcpy(input_array_d, input_array_h, R * C * Z * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(F, filter_h, (FILTER_RADIUS*2 + 1) * (FILTER_RADIUS*2 + 1)* sizeof(float));

   // run kernel
   gettimeofday(&t1, 0);
   dim3 dim_block(IN_TILE_DIM, IN_TILE_DIM, CHANNEL);
   dim3 dim_grid(ceil(C * 1.0f/OUT_TILE_DIM), ceil(R * 1.0f/OUT_TILE_DIM));
   conv2d_gpu<<<dim_grid, dim_block>>>(input_array_d, output_array_d, R, C);

   //6. Check error from kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    gpuErrchk(cudaDeviceSynchronize());

   // copy memory back to host
   cudaMemcpy(output_array_h_gpu, output_array_d, R * C * sizeof(float), cudaMemcpyDeviceToHost);
   gettimeofday(&t2, 0);
   time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   printf("GPU Time to generate:  %3.1f ms \n", time);

   // print array
   // print_array(input_array_h, R, C, Z);
   // printf("filter\n");
   // print_array(filter_h, (FILTER_RADIUS*2 + 1), (FILTER_RADIUS*2 + 1), 1);
   // printf("conv2d_cpu\n");
   // print_array(output_array_h, R, C, 1);

   // print_array(input_array_h, R, C, Z);
   // printf("filter\n");
   // print_array(filter_h, (FILTER_RADIUS*2 + 1), (FILTER_RADIUS*2 + 1), 1);
   // printf("conv2d_gpu\n");
   // print_array(output_array_h_gpu, R, C, 1);

   // free memory
   cudaFree(input_array_d);
   cudaFree(output_array_d);
   //cudaFree(F);
   free(input_array_h);
   free(output_array_h);
   free(output_array_h_gpu);
   free(filter_h);
}

// For cuda debugging -> nvcc -G -g mykernel.cu -o myapp