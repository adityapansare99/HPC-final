/**
 * ============================================================================
 * High Performance Computing (HPC) - Experiment No. 11
 * Title: CUDA Performance Optimization and cuBLAS
 * 
 * Task 2: Matrix Multiplication Using cuBLAS
 * - Step 1: Initialize host input matrices and sequential CPU reference
 * - Step 2: cuBLAS library handle creation and device buffer allocation
 * - Step 3: Handle Column-Major vs Row-Major layout via algebraic identity:
 *           C = A * B  <=>  C^T = B^T * A^T
 *           Passing d_B as first matrix and d_A as second matrix with leading
 *           dimensions N and K yields the exact row-major result d_C!
 * - Step 4: Library execution timing with cudaEvent vs End-to-End timing
 * - Step 5: Verification against CPU reference (Max Absolute Error & RMSE)
 * - Step 6: Speedup and GFLOPS calculation across N in {256, 512, 1024, 2048}
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status Code: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ----------------------------------------------------------------------------
// CPU Matrix Multiplication (i-k-j loop order)
// ----------------------------------------------------------------------------
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N * N; ++i) C[i] = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Basic CUDA Kernel for comparison
// ----------------------------------------------------------------------------
__global__ void matmul_basic_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Tiled CUDA Kernel (16x16) for comparison
// ----------------------------------------------------------------------------
template <int TILE_DIM>
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        int b_row = t * TILE_DIM + threadIdx.y;
        s_B[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Verification function
bool verify_results(const float* ref, const float* test, int size,
                    float& max_err, float& rmse, float tol = 1e-3f) {
    max_err = 0.0f;
    double sq_diff_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > max_err) max_err = diff;
        sq_diff_sum += diff * diff;
    }
    rmse = std::sqrt(sq_diff_sum / size);
    return (max_err < tol);
}

double calc_gflops(int N, double time_ms) {
    double operations = 2.0 * static_cast<double>(N) * N * N;
    return (operations / (time_ms * 1e-3)) / 1e9;
}

// ----------------------------------------------------------------------------
// Run Task 2 for Specified Size
// ----------------------------------------------------------------------------
void run_task2_for_size(int N, cublasHandle_t handle) {
    std::cout << "\n========================================================================\n";
    std::cout << "  TASK 2: cuBLAS vs CUSTOM CUDA: Size " << N << " x " << N << "\n";
    std::cout << "========================================================================\n";

    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N, 0.0f);
    std::vector<float> h_C_cublas(N * N, 0.0f);
    std::vector<float> h_C_tiled(N * N, 0.0f);
    std::vector<float> h_C_basic(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Allocate Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // 1. Basic CUDA Timing
    float basic_kernel_ms = 0.0f;
    {
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        matmul_basic_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_event));
        matmul_basic_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&basic_kernel_ms, start_event, stop_event));
        CUDA_CHECK(cudaMemcpy(h_C_basic.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    }

    // 2. Tiled CUDA (16x16) Timing
    float tiled_kernel_ms = 0.0f;
    {
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        matmul_tiled_kernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_event));
        matmul_tiled_kernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&tiled_kernel_ms, start_event, stop_event));
        CUDA_CHECK(cudaMemcpy(h_C_tiled.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    }

    // 3. cuBLAS Timing
    // cuBLAS uses column-major order:
    // C = A * B in row-major <=> C^T = B^T * A^T in column-major
    // Therefore:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float cublas_kernel_ms = 0.0f, cublas_h2d_ms = 0.0f, cublas_d2h_ms = 0.0f;

    // Warmup
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed H2D
    CUDA_CHECK(cudaEventRecord(start_event));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&cublas_h2d_ms, start_event, stop_event));

    // Timed cuBLAS Library Call
    CUDA_CHECK(cudaEventRecord(start_event));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&cublas_kernel_ms, start_event, stop_event));

    // Timed D2H
    CUDA_CHECK(cudaEventRecord(start_event));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&cublas_d2h_ms, start_event, stop_event));

    float cublas_total_ms = cublas_h2d_ms + cublas_kernel_ms + cublas_d2h_ms;

    // Verifications
    float err_basic = 0, rmse_basic = 0;
    float err_tiled = 0, rmse_tiled = 0;
    float err_cublas = 0, rmse_cublas = 0;
    bool pass_basic = verify_results(h_C_cpu.data(), h_C_basic.data(), N * N, err_basic, rmse_basic);
    bool pass_tiled = verify_results(h_C_cpu.data(), h_C_tiled.data(), N * N, err_tiled, rmse_tiled);
    bool pass_cublas = verify_results(h_C_cpu.data(), h_C_cublas.data(), N * N, err_cublas, rmse_cublas);

    // Performance Calculations
    double gflops_cpu = calc_gflops(N, cpu_time_ms);
    double gflops_basic = calc_gflops(N, basic_kernel_ms);
    double gflops_tiled = calc_gflops(N, tiled_kernel_ms);
    double gflops_cublas = calc_gflops(N, cublas_kernel_ms);

    double speedup_basic = cpu_time_ms / basic_kernel_ms;
    double speedup_tiled = cpu_time_ms / tiled_kernel_ms;
    double speedup_cublas = cpu_time_ms / cublas_kernel_ms;
    double cublas_vs_tiled = tiled_kernel_ms / cublas_kernel_ms;

    // Display Table as requested in Assignment PDF Task 2
    std::cout << "\nTask 2 Results Table (Matrix Size: " << N << " x " << N << "):\n";
    std::cout << std::string(96, '-') << "\n";
    std::cout << std::left << std::setw(26) << "Implementation"
              << std::right << std::setw(14) << "Time (ms)"
              << std::setw(14) << "GFLOPS"
              << std::setw(14) << "Speedup(CPU)"
              << std::setw(14) << "cuBLAS Gain"
              << std::setw(14) << "Status"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    std::cout << std::left << std::setw(26) << "Sequential CPU"
              << std::right << std::setw(14) << std::fixed << std::setprecision(3) << cpu_time_ms
              << std::setw(14) << std::setprecision(2) << gflops_cpu
              << std::setw(14) << "1.00x"
              << std::setw(14) << "-"
              << std::setw(14) << "BASELINE"
              << "\n";

    std::cout << std::left << std::setw(26) << "Basic CUDA (Global)"
              << std::right << std::setw(14) << std::fixed << std::setprecision(3) << basic_kernel_ms
              << std::setw(14) << std::setprecision(2) << gflops_basic
              << std::setw(14) << std::setprecision(2) << speedup_basic << "x"
              << std::setw(14) << std::setprecision(2) << (basic_kernel_ms / cublas_kernel_ms) << "x"
              << std::setw(14) << (pass_basic ? "PASSED" : "FAILED")
              << "\n";

    std::cout << std::left << std::setw(26) << "Tiled CUDA (16x16)"
              << std::right << std::setw(14) << std::fixed << std::setprecision(3) << tiled_kernel_ms
              << std::setw(14) << std::setprecision(2) << gflops_tiled
              << std::setw(14) << std::setprecision(2) << speedup_tiled << "x"
              << std::setw(14) << std::setprecision(2) << cublas_vs_tiled << "x"
              << std::setw(14) << (pass_tiled ? "PASSED" : "FAILED")
              << "\n";

    std::cout << std::left << std::setw(26) << "cuBLAS (cublasSgemm)"
              << std::right << std::setw(14) << std::fixed << std::setprecision(3) << cublas_kernel_ms
              << std::setw(14) << std::setprecision(2) << gflops_cublas
              << std::setw(14) << std::setprecision(2) << speedup_cublas << "x"
              << std::setw(14) << "1.00x"
              << std::setw(14) << (pass_cublas ? "PASSED" : "FAILED")
              << "\n";

    std::cout << std::string(96, '-') << "\n";
    std::cout << "cuBLAS End-to-End Breakdown: H2D=" << cublas_h2d_ms << " ms | SGEMM="
              << cublas_kernel_ms << " ms | D2H=" << cublas_d2h_ms << " ms | Total=" << cublas_total_ms << " ms\n";
    std::cout << "Numerical Verification: Max Absolute Error = " << std::scientific << err_cublas
              << ", RMSE = " << rmse_cublas << "\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << " HPC EXPERIMENT 11: TASK 2 - cuBLAS MATRIX MULTIPLICATION\n";
    std::cout << " Student: Aditya | PRN: 23510080\n";
    std::cout << "========================================================================\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {256, 512, 1024};
    for (int N : sizes) {
        run_task2_for_size(N, handle);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    std::cout << "\nTask 2 cuBLAS execution completed successfully!\n";
    return 0;
}
