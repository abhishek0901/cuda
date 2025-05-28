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

__host__ __device__ void merge_seq_circular(int *A, int *B, int *C, int size_A, int size_B, int A_S_start, int B_S_start, int tile_size) {
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < size_A) && (j < size_B)) {
        int i_cir = (i + A_S_start) % tile_size;
        int j_cir = (j + B_S_start) % tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir];i++;
        } else {
            C[k++] = B[j_cir];j++;
        }
    }

    if (i == size_A) {
        while(j < size_B) {
            int j_cir = (j + B_S_start) % tile_size;
            C[k++] = B[j_cir];j++;
        }
    } else {
        while (i < size_A) {
            int i_cir = (i + A_S_start) % tile_size;
            C[k++] = A[i_cir];i++;
        }
    }
}


__device__ int co_rank_circular(int k, int *A, int size_A, int *B, int size_B, int A_S_start, int B_S_start, int tile_size) {
    int i = k < size_A ? k : size_A;
    int j = k - i;
    int i_low = 0 > (k - size_B) ? 0 : k - size_B;
    int j_low = 0 > (k - size_A) ? 0 : k - size_A;
    int delta = 0;
    bool active = true;

    while (active) {
        int i_cir = (i + A_S_start) % tile_size;
        int i_m_1_cir = (i + A_S_start - 1) % tile_size;
        int j_cir = (j + B_S_start) % tile_size;
        int j_m_1_cir = (j + B_S_start - 1) % tile_size;
        if (i > 0 && j < size_B && A[i_m_1_cir] >= B[j_cir]) {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < size_A && B[j_m_1_cir] > A[i_cir]) {
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

__global__ void circular_tiled_merge_gpu(int *A, int *B, int *C, int size_A, int size_B, int tile_size) {
    extern __shared__ int sharedAB[];
    int *A_S = &sharedAB[0];
    int *B_S = &sharedAB[tile_size];

    int C_curr = blockIdx.x * ceil((size_A + size_B) * 1.0 / gridDim.x);
    int C_next = min((blockIdx.x + 1) * ceil((size_A + size_B) * 1.0 / gridDim.x), (size_A + size_B) * 1.0);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank_circular(C_curr, A, size_A, B, size_B, 0, 0, max(size_A, size_B) + 1);
        A_S[1] = co_rank_circular(C_next, A, size_A, B, size_B, 0, 0, max(size_A, size_B) + 1);
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil(C_length * 1.0 / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size; // amnt to fill in the array
    int B_S_consumed = tile_size;

    while (counter < total_iteration) {
        /*
            Load tile size A and B into Shared memory.
        */
       for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if ((i + threadIdx.x < A_length - A_consumed) && (i + threadIdx.x) < A_S_consumed) {
                A_S[(i + threadIdx.x + A_S_start + tile_size - A_S_consumed)%tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
            }            
       }
       for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if ((i + threadIdx.x < B_length - B_consumed) && (i + threadIdx.x) < B_S_consumed) {
                B_S[(i + threadIdx.x + B_S_start + tile_size - B_S_consumed)%tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
            }            
       }
       __syncthreads();
       int c_curr = threadIdx.x * (tile_size / blockDim.x);
       int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
       c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
       c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

       // Find co rank of c_curr and c_next
       int a_curr = co_rank_circular(c_curr, 
                            A_S, 
                            min(tile_size, A_length - A_consumed), 
                            B_S, 
                            min(tile_size, B_length - B_consumed),
                            A_S_start,
                            B_S_start,
                            tile_size
        );
        int a_next = co_rank_circular(c_next, 
                            A_S, 
                            min(tile_size, A_length - A_consumed), 
                            B_S, 
                            min(tile_size, B_length - B_consumed),
                            A_S_start,
                            B_S_start,
                            tile_size
        );
        int b_curr = c_curr - a_curr;
        int b_next = c_next - a_next;

        // Call seq merge function
        merge_seq_circular(A_S, B_S, C + C_curr + C_completed + c_curr, a_next - a_curr, b_next - b_curr, A_S_start + a_curr, B_S_start + b_curr, tile_size);
        counter++;
        A_S_consumed = co_rank_circular(
            min(tile_size, C_length - C_completed),
            A_S,min(tile_size, A_length - A_consumed),
            B_S, min(tile_size, B_length - B_consumed),
            A_S_start,B_S_start, tile_size
        );
        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;
        __syncthreads();
    }
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
    int tile_size = 1024;
    int sharedMemSize = 2 * tile_size * sizeof(int);
    cudaMalloc((void**)&A_d, size_A * sizeof(int));
    cudaMalloc((void**)&B_d, size_B * sizeof(int));
    cudaMalloc((void**)&C_d, (size_A + size_B) * sizeof(int));

    // Memcpy
    cudaMemcpy(A_d, A, size_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_B * sizeof(int), cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dim_block(NUM_THREADS);
    dim3 dim_grid(ceil((size_A + size_B) * 1.0/ (NUM_THREADS * 32)));
    circular_tiled_merge_gpu<<<dim_grid, dim_block, sharedMemSize>>>(A_d, B_d, C_d, size_A, size_B,tile_size);
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