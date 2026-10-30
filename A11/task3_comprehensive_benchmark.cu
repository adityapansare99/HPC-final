/**
 * ============================================================================
 * High Performance Computing (HPC) - Experiment No. 11
 * Title: CUDA Performance Optimization and cuBLAS
 * 
 * Task 3: Comprehensive Unified Benchmark & Performance Profiling
 * 
 * Benchmarks:
 * 1. Sequential CPU (optimized i-k-j loop order)
 * 2. Basic CUDA (Global Memory, 16x16 block)
 * 3. Tiled CUDA (Shared Memory, Tile 8x8)
 * 4. Tiled CUDA (Shared Memory, Tile 16x16)
 * 5. Tiled CUDA (Shared Memory, Tile 32x32)
 * 6. cuBLAS (cublasSgemm)
 * 
 * Matrix Sizes: 256x256, 512x512, 1024x1024 (and 2048x2048)
 * Outputs: Terminal table & machine-readable JSON for automatic graphing.
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
// Basic CUDA Kernel (Global Memory)
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
// Tiled CUDA Kernel (Shared Memory)
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

struct MethodResult {
    std::string key;
    std::string display_name;
    int tile_size;
    float kernel_ms;
    float h2d_ms;
    float d2h_ms;
    float total_ms;
    double gflops;
    double speedup_kernel;
    double speedup_total;
    float max_err;
    float rmse;
    bool passed;
};

struct SizeResult {
    int N;
    double cpu_time_ms;
    double cpu_gflops;
    std::vector<MethodResult> methods;
};

void run_benchmark_for_size(int N, cublasHandle_t handle, SizeResult& sr) {
    std::cout << "\n========================================================================\n";
    std::cout << ">>> BENCHMARKING SIZE " << N << " x " << N << " (" << (N * N * sizeof(float)) / (1024.0 * 1024.0) << " MB per matrix)\n";
    std::cout << "========================================================================\n";

    sr.N = N;
    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N, 0.0f);
    std::vector<float> h_C_gpu(N * N, 0.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // CPU execution
    std::cout << "--> Computing CPU reference..." << std::flush;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    sr.cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    sr.cpu_gflops = calc_gflops(N, sr.cpu_time_ms);
    std::cout << " Done in " << std::fixed << std::setprecision(2) << sr.cpu_time_ms
              << " ms (" << sr.cpu_gflops << " GFLOPS)\n";

    // Allocate Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    auto benchmark_gpu = [&](const std::string& key, const std::string& name, int tile, auto launcher) {
        // Warmup
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        launcher();
        CUDA_CHECK(cudaDeviceSynchronize());

        // H2D
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(start_event));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_event, stop_event));

        // Kernel (average over 5 runs)
        float total_kernel_time = 0.0f;
        int num_trials = 5;
        for (int it = 0; it < num_trials; ++it) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(start_event));
            launcher();
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
            total_kernel_time += ms;
        }
        float kernel_ms = total_kernel_time / num_trials;

        // D2H
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(start_event));
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_event, stop_event));

        float total_ms = h2d_ms + kernel_ms + d2h_ms;
        float max_err = 0.0f, rmse = 0.0f;
        bool passed = verify_results(h_C_cpu.data(), h_C_gpu.data(), N * N, max_err, rmse);

        MethodResult mr;
        mr.key = key;
        mr.display_name = name;
        mr.tile_size = tile;
        mr.kernel_ms = kernel_ms;
        mr.h2d_ms = h2d_ms;
        mr.d2h_ms = d2h_ms;
        mr.total_ms = total_ms;
        mr.gflops = calc_gflops(N, kernel_ms);
        mr.speedup_kernel = sr.cpu_time_ms / kernel_ms;
        mr.speedup_total = sr.cpu_time_ms / total_ms;
        mr.max_err = max_err;
        mr.rmse = rmse;
        mr.passed = passed;
        sr.methods.push_back(mr);
    };

    // 1. Basic CUDA (16x16)
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        benchmark_gpu("basic_cuda", "Basic CUDA (Global)", 16, [&]() {
            matmul_basic_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 2. Tiled CUDA 8x8
    {
        dim3 block(8, 8);
        dim3 grid((N + 7) / 8, (N + 7) / 8);
        benchmark_gpu("tiled_8x8", "Tiled CUDA (8x8)", 8, [&]() {
            matmul_tiled_kernel<8><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 3. Tiled CUDA 16x16
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        benchmark_gpu("tiled_16x16", "Tiled CUDA (16x16)", 16, [&]() {
            matmul_tiled_kernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 4. Tiled CUDA 32x32
    {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        benchmark_gpu("tiled_32x32", "Tiled CUDA (32x32)", 32, [&]() {
            matmul_tiled_kernel<32><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 5. cuBLAS
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        benchmark_gpu("cublas", "cuBLAS (cublasSgemm)", 0, [&]() {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        });
    }

    // Print Table
    std::cout << "\n" << std::string(98, '-') << "\n";
    std::cout << std::left << std::setw(24) << "Implementation"
              << std::right << std::setw(12) << "Kernel (ms)"
              << std::setw(12) << "H2D (ms)"
              << std::setw(12) << "D2H (ms)"
              << std::setw(12) << "Total (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(14) << "Speedup(K)"
              << "\n";
    std::cout << std::string(98, '-') << "\n";

    std::cout << std::left << std::setw(24) << "Sequential CPU"
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << sr.cpu_time_ms
              << std::setw(12) << "-"
              << std::setw(12) << "-"
              << std::setw(12) << sr.cpu_time_ms
              << std::setw(12) << std::setprecision(2) << sr.cpu_gflops
              << std::setw(14) << "1.00x"
              << "\n";

    for (const auto& mr : sr.methods) {
        std::cout << std::left << std::setw(24) << mr.display_name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << mr.kernel_ms
                  << std::setw(12) << mr.h2d_ms
                  << std::setw(12) << mr.d2h_ms
                  << std::setw(12) << mr.total_ms
                  << std::setw(12) << std::setprecision(1) << mr.gflops
                  << std::setw(13) << std::setprecision(2) << mr.speedup_kernel << "x"
                  << "  [" << (mr.passed ? "PASS" : "FAIL") << "]"
                  << "\n";
    }
    std::cout << std::string(98, '-') << "\n";

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << " HPC EXPERIMENT 11: TASK 3 - COMPREHENSIVE UNIFIED BENCHMARK\n";
    std::cout << " CUDA Performance Optimization & cuBLAS Profiling\n";
    std::cout << " Student: Aditya | PRN: 23510080\n";
    std::cout << "========================================================================\n";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (" << prop.totalGlobalMem / (1024*1024) << " MB)\n";
    std::cout << "CUDA Cores Architecture: Ampere SM " << prop.major << "." << prop.minor << "\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {256, 512, 1024};
    if (argc > 1 && std::string(argv[1]) == "--include-2048") {
        sizes.push_back(2048);
    }

    std::vector<SizeResult> all_results;
    for (int N : sizes) {
        SizeResult sr;
        run_benchmark_for_size(N, handle, sr);
        all_results.push_back(sr);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    // Print JSON payload for python automation
    std::cout << "\n---JSON_START---\n{\n";
    std::cout << "  \"device\": \"" << prop.name << "\",\n";
    std::cout << "  \"cuda_version\": \"12.4 / 13.4\",\n";
    std::cout << "  \"benchmarks\": [\n";
    for (size_t i = 0; i < all_results.size(); ++i) {
        const auto& sr = all_results[i];
        std::cout << "    {\n";
        std::cout << "      \"matrix_size\": " << sr.N << ",\n";
        std::cout << "      \"cpu_time_ms\": " << sr.cpu_time_ms << ",\n";
        std::cout << "      \"cpu_gflops\": " << sr.cpu_gflops << ",\n";
        std::cout << "      \"methods\": [\n";
        for (size_t j = 0; j < sr.methods.size(); ++j) {
            const auto& m = sr.methods[j];
            std::cout << "        {\n";
            std::cout << "          \"key\": \"" << m.key << "\",\n";
            std::cout << "          \"name\": \"" << m.display_name << "\",\n";
            std::cout << "          \"tile_size\": " << m.tile_size << ",\n";
            std::cout << "          \"kernel_ms\": " << m.kernel_ms << ",\n";
            std::cout << "          \"h2d_ms\": " << m.h2d_ms << ",\n";
            std::cout << "          \"d2h_ms\": " << m.d2h_ms << ",\n";
            std::cout << "          \"total_ms\": " << m.total_ms << ",\n";
            std::cout << "          \"gflops\": " << m.gflops << ",\n";
            std::cout << "          \"speedup_kernel\": " << m.speedup_kernel << ",\n";
            std::cout << "          \"speedup_total\": " << m.speedup_total << ",\n";
            std::cout << "          \"max_err\": " << m.max_err << ",\n";
            std::cout << "          \"rmse\": " << m.rmse << ",\n";
            std::cout << "          \"passed\": " << (m.passed ? "true" : "false") << "\n";
            std::cout << "        }" << (j + 1 < sr.methods.size() ? "," : "") << "\n";
        }
        std::cout << "      ]\n";
        std::cout << "    }" << (i + 1 < all_results.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n}\n---JSON_END---\n";

    return 0;
}
