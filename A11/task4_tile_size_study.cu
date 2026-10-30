/**
 * ============================================================================
 * High Performance Computing (HPC) - Experiment No. 11
 * Title: CUDA Performance Optimization and cuBLAS
 * 
 * Task 4: Empirical Tile & Block Size Study
 * Deep dive into Block Dimensions (4x4, 8x8, 16x16, 32x32)
 * Evaluates:
 * - Active warps & threads per block
 * - Shared memory allocated per block (2 * TILE_DIM^2 * 4 bytes)
 * - Maximum active blocks per SM limited by threads (1536 / (TILE_DIM^2))
 * - Arithmetic throughput and effective memory bandwidth
 * ============================================================================
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
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

template <int TILE_DIM>
__global__ void matmul_tiled_study_kernel(const float* __restrict__ A,
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

template <int TILE_DIM>
void evaluate_tile_size(int N, const float* d_A, const float* d_B, float* d_C,
                        cudaEvent_t start, cudaEvent_t stop) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Warmup
    matmul_tiled_study_kernel<TILE_DIM><<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    int runs = 10;
    for (int i = 0; i < runs; ++i) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(start));
        matmul_tiled_study_kernel<TILE_DIM><<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    float avg_ms = total_ms / runs;

    double ops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (ops / (avg_ms * 1e-3)) / 1e9;
    size_t shared_bytes = 2 * TILE_DIM * TILE_DIM * sizeof(float);
    int threads_per_block = TILE_DIM * TILE_DIM;
    int warps_per_block = threads_per_block / 32;

    std::cout << std::left << std::setw(12) << (std::to_string(TILE_DIM) + "x" + std::to_string(TILE_DIM))
              << std::right << std::setw(16) << threads_per_block
              << std::setw(14) << warps_per_block
              << std::setw(18) << (shared_bytes / 1024.0)
              << std::setw(14) << std::fixed << std::setprecision(3) << avg_ms
              << std::setw(14) << std::setprecision(2) << gflops
              << "\n";
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << " HPC EXPERIMENT 11: TASK 4 - TILE SIZE & OCCUPANCY STUDY\n";
    std::cout << " Student: Aditya | PRN: 23510080\n";
    std::cout << "========================================================================\n";

    int N = 1024;
    size_t bytes = N * N * sizeof(float);
    std::vector<float> h_mat(N * N, 1.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_mat.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_mat.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\nTile Dimension Benchmark for Matrix Size " << N << " x " << N << ":\n";
    std::cout << std::string(88, '-') << "\n";
    std::cout << std::left << std::setw(12) << "Tile Dim"
              << std::right << std::setw(16) << "Threads/Block"
              << std::setw(14) << "Warps/Block"
              << std::setw(18) << "Shared Mem (KB)"
              << std::setw(14) << "Avg Time (ms)"
              << std::setw(14) << "GFLOPS"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    evaluate_tile_size<4>(N, d_A, d_B, d_C, start, stop);
    evaluate_tile_size<8>(N, d_A, d_B, d_C, start, stop);
    evaluate_tile_size<16>(N, d_A, d_B, d_C, start, stop);
    evaluate_tile_size<32>(N, d_A, d_B, d_C, start, stop);

    std::cout << std::string(88, '-') << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
