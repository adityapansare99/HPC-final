/**
 * ============================================================================
 * High Performance Computing (HPC) - Experiment No. 11
 * Title: CUDA Performance Optimization and cuBLAS
 * 
 * Task 1: Optimize CUDA Matrix Multiplication
 * - Step 1: Sequential CPU implementation (cache-friendly i-k-j loop order)
 * - Step 2: Basic CUDA implementation (1 thread per output element, global memory)
 * - Step 3: Optimized CUDA implementation (Shared Memory Tiling: 8x8, 16x16, 32x32)
 * 
 * Features:
 * - Rigorous verification against CPU reference (Max Absolute Error, RMSE)
 * - Precise CUDA event timing for kernel-only execution
 * - Host-to-Device and Device-to-Host transfer timing breakdown
 * - Speedup calculation and GFLOPS computation
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ----------------------------------------------------------------------------
// Step 1: Sequential CPU Matrix Multiplication
// Using i-k-j loop order for optimal cache spatial locality (stride-1 access)
// ----------------------------------------------------------------------------
void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N * N; ++i) {
        C[i] = 0.0f;
    }
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
// Step 2: Basic CUDA Matrix Multiplication (Global Memory)
// Each thread computes one element of C.
// Suffers from high memory latency: 2*N global memory reads per thread.
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
// Step 3: Optimized CUDA Matrix Multiplication (Shared Memory Tiling)
// Data is loaded in TILE_DIM x TILE_DIM chunks into high-speed on-chip shared memory.
// Threads synchronize via __syncthreads() to avoid data hazards.
// Data is reused TILE_DIM times, reducing off-chip global memory bandwidth by TILE_DIM.
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
        // Load tile from matrix A into shared memory
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < N && a_col < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from matrix B into shared memory
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < N && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Barrier: Ensure all threads in the block have finished loading
        __syncthreads();

        // Compute partial dot product from shared memory tiles
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Barrier: Ensure all threads have finished computation before loading next tile
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ----------------------------------------------------------------------------
// Numerical Verification Utility
// Calculates maximum absolute error and root mean squared error (RMSE)
// ----------------------------------------------------------------------------
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

// Helper: Calculate GFLOPS
double calc_gflops(int N, double time_ms) {
    // 2 * N^3 operations: N^3 multiplications + N^3 additions
    double operations = 2.0 * static_cast<double>(N) * N * N;
    return (operations / (time_ms * 1e-3)) / 1e9;
}

// ----------------------------------------------------------------------------
// Execution Wrapper for Given Matrix Size
// ----------------------------------------------------------------------------
void run_task1_for_size(int N) {
    std::cout << "\n========================================================================\n";
    std::cout << "  BENCHMARKING MATRIX MULTIPLICATION: Size " << N << " x " << N << "\n";
    std::cout << "  Total Elements: " << N * N << " (" << (N * N * sizeof(float)) / (1024.0 * 1024.0) << " MB per matrix)\n";
    std::cout << "========================================================================\n";

    size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N, 0.0f);
    std::vector<float> h_C_gpu(N * N, 0.0f);

    // Initialize with deterministic pseudorandom data in [0, 1]
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // 1. Sequential CPU
    std::cout << "--> Running Sequential CPU Matrix Multiplication..." << std::flush;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_gflops = calc_gflops(N, cpu_time_ms);
    std::cout << " Done! CPU Time: " << std::fixed << std::setprecision(3) << cpu_time_ms
              << " ms (" << cpu_gflops << " GFLOPS)\n";

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    // Structure to hold results
    struct BenchmarkRecord {
        std::string name;
        int tile_dim;
        float kernel_ms;
        float h2d_ms;
        float d2h_ms;
        float total_ms;
        double gflops_kernel;
        double speedup_kernel;
        double speedup_total;
        float max_err;
        float rmse;
        bool passed;
    };
    std::vector<BenchmarkRecord> records;

    // Lambda to benchmark a CUDA kernel
    auto benchmark_kernel = [&](const std::string& name, int tile_dim, auto kernel_launcher) {
        float max_err = 0.0f, rmse = 0.0f;
        float kernel_ms = 0.0f, h2d_ms = 0.0f, d2h_ms = 0.0f;

        // Warmup run
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        kernel_launcher();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run: H2D
        CUDA_CHECK(cudaEventRecord(start_event));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start_event, stop_event));

        // Timed run: Kernel
        CUDA_CHECK(cudaEventRecord(start_event));
        kernel_launcher();
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));

        // Timed run: D2H
        CUDA_CHECK(cudaEventRecord(start_event));
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start_event, stop_event));

        float total_ms = h2d_ms + kernel_ms + d2h_ms;
        bool passed = verify_results(h_C_cpu.data(), h_C_gpu.data(), N * N, max_err, rmse);

        records.push_back({
            name,
            tile_dim,
            kernel_ms,
            h2d_ms,
            d2h_ms,
            total_ms,
            calc_gflops(N, kernel_ms),
            cpu_time_ms / kernel_ms,
            cpu_time_ms / total_ms,
            max_err,
            rmse,
            passed
        });
    };

    // 2. Basic CUDA (Tile 16x16)
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        benchmark_kernel("Basic CUDA (Global)", 16, [&]() {
            matmul_basic_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 3. Tiled CUDA: 8x8
    {
        dim3 block(8, 8);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        benchmark_kernel("Tiled CUDA (8x8)", 8, [&]() {
            matmul_tiled_kernel<8><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 4. Tiled CUDA: 16x16
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        benchmark_kernel("Tiled CUDA (16x16)", 16, [&]() {
            matmul_tiled_kernel<16><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // 5. Tiled CUDA: 32x32 (1024 threads per block)
    {
        dim3 block(32, 32);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        benchmark_kernel("Tiled CUDA (32x32)", 32, [&]() {
            matmul_tiled_kernel<32><<<grid, block>>>(d_A, d_B, d_C, N);
        });
    }

    // Output formatted results table
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
              << std::right << std::setw(12) << cpu_time_ms
              << std::setw(12) << "-"
              << std::setw(12) << "-"
              << std::setw(12) << cpu_time_ms
              << std::setw(12) << cpu_gflops
              << std::setw(14) << "1.00x"
              << "\n";

    for (const auto& r : records) {
        std::cout << std::left << std::setw(24) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.kernel_ms
                  << std::setw(12) << r.h2d_ms
                  << std::setw(12) << r.d2h_ms
                  << std::setw(12) << r.total_ms
                  << std::setw(12) << std::setprecision(1) << r.gflops_kernel
                  << std::setw(13) << std::setprecision(2) << r.speedup_kernel << "x"
                  << "  [" << (r.passed ? "PASS" : "FAIL") << "]"
                  << "\n";
    }
    std::cout << std::string(98, '-') << "\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << " HPC EXPERIMENT 11: TASK 1 - OPTIMIZE CUDA MATRIX MULTIPLICATION\n";
    std::cout << " Student: Aditya | PRN: 23510080\n";
    std::cout << "========================================================================\n";

    // Print Device Properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";

    // Test matrix sizes specified in PDF: 256, 512, 1024
    std::vector<int> sizes = {256, 512, 1024};
    for (int N : sizes) {
        run_task1_for_size(N);
    }

    std::cout << "\nTask 1 execution completed successfully!\n";
    return 0;
}
