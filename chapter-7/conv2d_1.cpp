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

// Define constant Memory
#define FILTER_RADIUS 1
__constant__ float F[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];

void init_array(float *input, int R, int C, int Z) {
   for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
         for (int z = 0; z < Z; z++) {
            //input[r * C * Z + c * Z + z] = (float) rand() / RAND_MAX * 255;
            input[r * C * Z + c * Z + z] = 1;
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
            printf("%f ", A[r * C * Z + c * Z + z]);
         }
         printf("\n");
      }
      printf("\n\n");
   }
}

void conv2d_cpu(float *input, float *filter, float *output, int R, int C, int Z, int width) {
   for (int r = 0; r < R; r++) {
      for (int c = 0; c < C; c++) {
         if (r==2 && c==0) {
            int test = 1;
         }
         int elem = 0;
         for (int z = 0; z < Z; z++) {
            for (int filter_r = -width; filter_r < width + 1; filter_r++) {
               for (int filter_c = -width; filter_c < width + 1; filter_c++) {
                  int input_row = r + filter_r, input_col = c + filter_c;
                  int filter_elem = filter[(filter_r + width) * (2*width+1) + (filter_c + width)];
                  if (input_row >= 0 && input_row < R && input_col >= 0 && input_col < C) {
                     int input_elem = input[input_row * C * Z + input_col * Z + z];
                     elem += filter_elem * input_elem;
                  }
               }
            }
         }
         output[r * C + c] = elem;
      }
   }
}

int main() {
   // define vars
   float *input_array_h, *filter_h, *output_array_h;

   //allocate memory -> A X B X C
   int R = 3, C = 3, Z= 1;
   input_array_h = (float*)malloc(R * C * Z * sizeof(float));
   output_array_h = (float*)malloc(R * C * sizeof(float));

   // define filter_h -> 3 X 3
   filter_h = (float*)malloc((FILTER_RADIUS*2 + 1) * (FILTER_RADIUS*2 + 1)* sizeof(float));

   // init array
   init_array(input_array_h, R, C, Z);

   // init filter
   init_filter(filter_h, (FILTER_RADIUS*2 + 1), (FILTER_RADIUS*2 + 1));

   // cpu conv2d
   conv2d_cpu(input_array_h, filter_h, output_array_h, R, C, Z, FILTER_RADIUS);

   // print array
   print_array(input_array_h, R, C, Z);
   printf("filter\n");
   print_array(filter_h, (FILTER_RADIUS*2 + 1), (FILTER_RADIUS*2 + 1), 1);
   printf("conv2d_cpu\n");
   print_array(output_array_h, R, C, 1);

   // free memory
   free(input_array_h);
   free(filter_h);
    
}