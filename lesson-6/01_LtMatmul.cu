#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call) {\
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

void cpu_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << "Matrix " << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    const int M = 4, K = 4, N = 4;

    // Manually define input matrices
    float h_A[M*K] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,
    };

    float h_B[K * N] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 11.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        14.0f, 17.0f, 25.0f, 36.0f,
    };

    float h_C_cpu[M*N] = {0};
    float h_C_gpu_fp32[M * N] = {0};
    float h_C_gpu_fp16[M * N] = {0};

    // Print input matrices
    print_matrix(h_A, M, K, "A");
    print_matrix(h_B, K, N , "B");

    // Allocate device memory to fp32
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

    // Allocate device memory to fp16
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));

    // Copy data to device(FP32)
    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice)); 

    // Convert and copy data to device(FP16)
    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    for (int i = 0; i < M * K; i++)h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++)h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Setup matrix Layout for FP32
    cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N));
    // Setup matrix Layout for FP16
    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N));

    // Setup matrix multiplication descriptor for FP32
    cublasLtMatmulDesc_t matmulDesc_fp32;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Setup matrix multiplication descriptor for FP16
    cublasLtMatmulDesc_t matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    // Update transpose attribute to mat mul fp32 desc and mat mul fp16 desc
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));


    // Perform matmul for fp32
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp32, &alpha, d_B_fp32, matB_fp32, d_A_fp32, matA_fp32, &beta, d_C_fp32, matC_fp32, d_C_fp32, matC_fp32, nullptr, nullptr, 0, 0));

    // Perform matmul for fp16
    const half alpha_half = __float2half(1.0f);
    const half beta_half = __float2half(0.0f);
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc_fp16, &alpha_half, d_B_fp16, matB_fp16, d_A_fp16, matA_fp16, &beta_half, d_C_fp16, matC_fp16, d_C_fp16, matC_fp16, nullptr, nullptr, 0, 0));

    // Copy fp32 to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy fp16 to host
    std::vector<half> h_C_fp16_gpu_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_fp16_gpu_half.data(), d_C_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Convert half precision to float
    for (int i = 0; i < M * N; i++)h_C_gpu_fp16[i] = __half2float(h_C_fp16_gpu_half[i]);

    // Do matmul on cpu
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);

    //Print Results
    print_matrix(h_C_cpu, M , N, "CPU");
    print_matrix(h_C_gpu_fp32, M , N, "GPU - F32");
    print_matrix(h_C_gpu_fp16, M , N, "GPU - F16");

    // Check if fp32, fp16 and cpu results match
    bool res_fp32 = true;
    bool res_fp16 = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C_gpu_fp32[i] - h_C_cpu[i]) > 1e-5) res_fp32 = false;
        if (fabs(h_C_gpu_fp16[i] - h_C_cpu[i]) > 1e-5) res_fp32 = false;
    }

    printf("FP32 Results Match : %d\n", res_fp32);
    printf("FP16 Results Match : %d\n", res_fp16);

    // Destroy objects and free memory
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp16));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    return 0;

    // nvcc -o res 01_LtMatmul.cu -lcublas -lcublasLt
}