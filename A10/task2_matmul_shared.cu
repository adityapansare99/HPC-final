/*
 * ==============================================================================
 * HPC Experiment 10 - Task 2: Tiled Matrix Multiplication Using Shared Memory
 * ==============================================================================

 * Description:
 *   Implements Tiled Matrix Multiplication using Shared Memory to exploit
 *   data reuse and memory locality:
 *     __shared__ float tileA[TILE][TILE];
 *     __shared__ float tileB[TILE][TILE];
 *   Algorithm:
 *     1. Load tile of A into shared memory.
 *     2. Load tile of B into shared memory.
 *     3. __syncthreads();
 *     4. Perform partial multiplication.
 *     5. __syncthreads();
 *     6. Move to next tile.
 *   Experiments across TILE = 8, 16, 32 and matrix sizes 256, 512, 1024.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Sequential CPU Matrix Multiplication */
void matmulCPU(const float *A, const float *B, float *C, int N) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[r * N + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

/* Templated Tiled Matrix Multiplication Kernel */
template <int TILE>
__global__ void matmulSharedKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int m = 0; m < numTiles; ++m) {
        // Collaborative loading of tileA into shared memory
        int aRow = row;
        int aCol = m * TILE + tx;
        if (aRow < N && aCol < N) {
            tileA[ty][tx] = A[aRow * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Collaborative loading of tileB into shared memory
        int bRow = m * TILE + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            tileB[ty][tx] = B[bRow * N + bCol];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize so all threads in block see loaded tiles
        __syncthreads();

        // Compute partial dot-product from fast on-chip shared memory
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize before next tile overwrites shared memory
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/* Initialize matrices */
void initMatrix(float *M, int N, float scale) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = sinf((float)i * 0.005f) * scale;
    }
}

/* Verify result */
bool verifyMatrix(const float *C_cpu, const float *C_gpu, int N, float tol = 1e-3f) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        float diff = fabsf(C_cpu[i] - C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            printf("  [Verification Failed] at [%d,%d]: CPU=%.5f, GPU=%.5f, Diff=%.5e\n",
                   i / N, i % N, C_cpu[i], C_gpu[i], diff);
            return false;
        }
    }
    printf("  [Verification Passed] Max Absolute Diff: %.5e\n", max_diff);
    return true;
}

template <int TILE>
void launchSharedKernel(const float *d_A, const float *d_B, float *d_C, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulSharedKernel<TILE><<<grid, block>>>(d_A, d_B, d_C, N);
}

void benchmarkSharedMatMul(int N, int tileSize) {
    printf("\n======================================================\n");
    printf(" Matrix Size: %d x %d | TILE Size: %d x %d (Shared Memory)\n", N, N, tileSize, tileSize);
    printf("======================================================\n");

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 0.5f);

    /* 1. CPU Reference */
    printf("  Executing CPU Sequential...");
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A, h_B, h_C_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf(" Done (%.2f ms)\n", cpu_ms);

    /* 2. GPU Allocation & Transfer */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t start_total, stop_total, start_kern, stop_kern;
    CHECK_CUDA(cudaEventCreate(&start_total));
    CHECK_CUDA(cudaEventCreate(&stop_total));
    CHECK_CUDA(cudaEventCreate(&start_kern));
    CHECK_CUDA(cudaEventCreate(&stop_kern));

    CHECK_CUDA(cudaEventRecord(start_total));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start_kern));
    if (tileSize == 8) {
        launchSharedKernel<8>(d_A, d_B, d_C, N);
    } else if (tileSize == 16) {
        launchSharedKernel<16>(d_A, d_B, d_C, N);
    } else if (tileSize == 32) {
        launchSharedKernel<32>(d_A, d_B, d_C, N);
    } else {
        fprintf(stderr, "Unsupported tile size: %d\n", tileSize);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaEventRecord(stop_kern));

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_total));
    CHECK_CUDA(cudaEventSynchronize(stop_total));

    float kernel_ms = 0.0f, total_gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kern, stop_kern));
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, start_total, stop_total));

    /* Verify correctness */
    verifyMatrix(h_C_cpu, h_C_gpu, N);

    /* Compute FLOPs */
    double gflops_kern = (2.0 * (double)N * N * N) / (kernel_ms * 1e6);
    double gflops_total = (2.0 * (double)N * N * N) / (total_gpu_ms * 1e6);
    double speedup_kern = cpu_ms / kernel_ms;
    double speedup_total = cpu_ms / total_gpu_ms;

    printf("  CPU Execution Time      : %10.3f ms\n", cpu_ms);
    printf("  CUDA Kernel Time        : %10.3f ms\n", kernel_ms);
    printf("  CUDA Total GPU Time     : %10.3f ms\n", total_gpu_ms);
    printf("  Kernel-Only Speedup     : %10.2fx\n", speedup_kern);
    printf("  End-to-End Speedup      : %10.2fx\n", speedup_total);
    printf("  Compute Throughput (Kern): %9.2f GFLOPS\n", gflops_kern);
    printf("  Throughput (End-to-End) : %9.2f GFLOPS\n", gflops_total);

    /* Output JSON snippet */
    printf("RESULT_JSON:{\"type\":\"matmul_shared\",\"size\":%d,\"tile\":%d,\"cpu_ms\":%.4f,\"kernel_ms\":%.4f,\"total_gpu_ms\":%.4f,\"speedup_kernel\":%.2f,\"speedup_total\":%.2f,\"gflops\":%.2f}\n",
           N, tileSize, cpu_ms, kernel_ms, total_gpu_ms, speedup_kern, speedup_total, gflops_kern);

    /* Cleanup */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start_total));
    CHECK_CUDA(cudaEventDestroy(stop_total));
    CHECK_CUDA(cudaEventDestroy(start_kern));
    CHECK_CUDA(cudaEventDestroy(stop_kern));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
}

int main(int argc, char **argv) {
    printf("====================================================================\n");
    printf(" CUDA Task 2: Tiled Matrix Multiplication Using Shared Memory\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("====================================================================\n");

    int test_sizes[] = {256, 512, 1024};
    int tile_sizes[] = {8, 16, 32};

    if (argc >= 3) {
        int custom_n = atoi(argv[1]);
        int custom_t = atoi(argv[2]);
        benchmarkSharedMatMul(custom_n, custom_t);
        return 0;
    }

    for (int s = 0; s < 3; ++s) {
        for (int t = 0; t < 3; ++t) {
            benchmarkSharedMatMul(test_sizes[s], tile_sizes[t]);
        }
    }

    return 0;
}
