#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <numeric>


#define CHECK_CUDA(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at line " << __LINE__ << ": " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


const int M = 4096;
const int K = 1024;
const int N = 4096;


// Naive cuda kernel for matrix multiplication
__global__ void naiveMatrixMultiply(const float *A, const float *B, float *C, int M, int K, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


// Function to initiate matrxi with random values
void initializeMatric(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < rows * cols; i++) matrix[i] = static_cast<float> (dis(gen));
}


bool verifyResults(const std::vector<float>& expected,const std::vector<float>& actual, float tolerance=1e-2) {
    if (expected.size() != actual.size()) {
        return false;
    }

    for (size_t i = 0; i < expected.size(); i++) {
        float rel_error = fabs(expected[i] - actual[i]);
        if (rel_error > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected " << expected[i] 
                << ", but got " << actual[i] << ", rel error " << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

template <typename Functor>
float time_kernel(Functor kernel_func) {
    cudaEvent_t start, stop;
    float elapsed_time;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_time;
}

template <typename Functor>
float benchmark_kernel(Functor kernel_func, int warmup_runs, int benchmark_runs) {
    for (int i = 0; i < warmup_runs; i++) kernel_func();

    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; i++) {
        times.push_back((float)time_kernel(kernel_func));
    }

    // Calculate avg time
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
    return avg_time;
}


int main() {
    std::cout << "Matrix A : (" << M << ", " << K << "), Matrix B : (" << K << ", " << N << ")" << std::endl;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    std::vector<float> h_C_cublas_fp32(M * N), h_C_cublasLt_fp32(M * N);
    std::vector<float> h_C_cublas_fp16(M * N), h_C_cublasLt_fp16(M * N);
    std::vector<float> h_C_naive(M * N);

    initializeMatric(h_A, M , K);
    initializeMatric(h_B, K, N);

    float *d_A, *d_B, *d_C;
    half *half_d_A, *half_d_B, *half_d_C;

    // Allocate Memory to device
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&half_d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&half_d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&half_d_C, M * N * sizeof(half)));


    // Copy memory to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<half> temp_half_A(M * K), temp_half_B(K * N);
    for (int i = 0; i < M * K; i++) temp_half_A[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) temp_half_B[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(half_d_A, temp_half_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(half_d_B, temp_half_B.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

    // Create cublas handle
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // Create cublasLt handle
    cublasLtHandle_t cublasLt_handle;
    CHECK_CUBLAS(cublasLtCreate(&cublasLt_handle));

    // Define alpha and warmup runs
    float alpha = 1.0f, beta = 0.0f;
    half half_alpha = __float2half(1.0f), half_beta = __float2half(0.0f);
    const int warmup_runs = 3, banchmark_runs = 20;

    // Run cubLas fp32
    float cublas_time_fp32 = benchmark_kernel(
        [&](){
            CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        },
        warmup_runs,
        banchmark_runs
    );
    std::cout << "cublas fp32 avg time " << cublas_time_fp32 << std::endl;
    CHECK_CUDA(cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Run cubLasLt fp32
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Alayout = nullptr, Blayout = nullptr, Clayout = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Alayout, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Blayout, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Clayout, CUDA_R_32F, N, M, N));

    float cublasLt_time_fp32 = benchmark_kernel(
        [&]() {
            CHECK_CUBLAS(cublasLtMatmul(cublasLt_handle, operationDesc, &alpha, d_B, Blayout, d_A, Alayout, &beta, d_C, Clayout, d_C, Clayout, nullptr, nullptr, 0, 0));
        },
        warmup_runs,
        banchmark_runs
    );
    std::cout << "cublasLt fp32 avg time " << cublasLt_time_fp32 << std::endl;
    CHECK_CUDA(cudaMemcpy(h_C_cublasLt_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Run cublas fp16
    float cublas_time_fp16 = benchmark_kernel(
        [&]() {
            CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &half_alpha, half_d_B, N, half_d_A, K, &half_beta, half_d_C, N));
        },
        warmup_runs,
        banchmark_runs
    );
    std::cout << "cublas fp16 avg time " << cublas_time_fp16 << std::endl;
    std::vector<half> half_cublas_C(M*N);
    CHECK_CUDA(cudaMemcpy(half_cublas_C.data(), half_d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) h_C_cublas_fp16[i] = __half2float(half_cublas_C[i]);

    // Run cublasLt fp16
    cublasLtMatmulDesc_t half_operationDesc = nullptr;
    cublasLtMatrixLayout_t half_Alayout = nullptr, half_Blayout = nullptr, half_Clayout = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&half_operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&half_Alayout, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&half_Blayout, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&half_Clayout, CUDA_R_16F, N, M, N));

    float cublasLt_time_fp16 = benchmark_kernel(
        [&]() {
            CHECK_CUBLAS(cublasLtMatmul(cublasLt_handle, half_operationDesc, &half_alpha, half_d_B, half_Blayout, half_d_A, half_Alayout, &beta, half_d_C, half_Clayout, half_d_C, half_Clayout, nullptr, nullptr, 0, 0));
        },
        warmup_runs,
        banchmark_runs
    );
    std::cout << "cublasLt fp16 avg time " << cublasLt_time_fp16 << std::endl;
    CHECK_CUDA(cudaMemcpy(half_cublas_C.data(), half_d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) h_C_cublasLt_fp16[i] = __half2float(half_cublas_C[i]);

    // Run naive kernel
    dim3 blockDim(32,32);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    float naive_cuda_time = benchmark_kernel(
        [&]() {
            naiveMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        },
        warmup_runs,
        banchmark_runs
    );
    std::cout << "naive cuda avg time " << naive_cuda_time << std::endl;
    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));


    // Verify Results
    bool cublas_fp32_correct = verifyResults(h_C_naive, h_C_cublas_fp32, 5e-1);
    bool cublas_fp16_correct = verifyResults(h_C_naive, h_C_cublas_fp16, 5e-1);
    bool cublasLt_fp32_correct = verifyResults(h_C_naive, h_C_cublasLt_fp32, 5e-1);
    bool cublasLt_fp16_correct = verifyResults(h_C_naive, h_C_cublasLt_fp16, 5e-1);

    std::cout << "cublas_fp32_correct : " << cublas_fp32_correct << std::endl;
    std::cout << "cublas_fp16_correct : " << cublas_fp16_correct << std::endl;
    std::cout << "cublasLt_fp32_correct : " << cublasLt_fp32_correct << std::endl;
    std::cout << "cublasLt_fp16_correct : " << cublasLt_fp16_correct << std::endl;

    // CLeanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(half_d_A));
    CHECK_CUDA(cudaFree(half_d_B));
    CHECK_CUDA(cudaFree(half_d_C));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUBLAS(cublasLtDestroy(cublasLt_handle));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Alayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Blayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Clayout));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(half_operationDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(half_Alayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(half_Blayout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(half_Clayout));
}