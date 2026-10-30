/*
 * ==============================================================================
 * HPC Experiment 10 - Task 1: Matrix Multiplication Using Global Memory
 * ==============================================================================

 * Description:
 *   Implements matrix multiplication C = A x B (N x N matrices):
 *     Version 1: Sequential CPU standard C/C++.
 *     Version 2: CUDA Global Memory where each thread computes one element of C.
 *   Thread indexing:
 *     int row = blockIdx.y * blockDim.y + threadIdx.y;
 *     int col = blockIdx.x * blockDim.x + threadIdx.x;
 *   Experiments across matrix sizes: 256x256, 512x512, 1024x1024
 *   and block configurations: 8x8, 16x16, 32x32.
 *   Measures CPU time, CUDA kernel time, CUDA total time, Speedup, GFLOPS.
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

/* CUDA Global Memory Matrix Multiplication Kernel */
__global__ void matmulGlobalKernel(const float *A, const float *B, float *C, int N) {
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

/* Initialize matrices with deterministic values */
void initMatrix(float *M, int N, float scale) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = sinf((float)i * 0.005f) * scale;
    }
}

/* Verify GPU result with CPU reference */
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

void benchmarkGlobalMatMul(int N, int blockSize) {
    printf("\n======================================================\n");
    printf(" Matrix Size: %d x %d | Block Configuration: %d x %d\n", N, N, blockSize, blockSize);
    printf("======================================================\n");

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 0.5f);

    /* 1. CPU Execution */
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

    dim3 block(blockSize, blockSize);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    CHECK_CUDA(cudaEventRecord(start_kern));
    matmulGlobalKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop_kern));

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_total));
    CHECK_CUDA(cudaEventSynchronize(stop_total));

    float kernel_ms = 0.0f, total_gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kern, stop_kern));
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, start_total, stop_total));

    /* Verify correctness */
    verifyMatrix(h_C_cpu, h_C_gpu, N);

    /* Compute FLOPs: 2 * N^3 operations */
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

    /* Output JSON snippet for runner script */
    printf("RESULT_JSON:{\"type\":\"matmul_global\",\"size\":%d,\"block\":%d,\"cpu_ms\":%.4f,\"kernel_ms\":%.4f,\"total_gpu_ms\":%.4f,\"speedup_kernel\":%.2f,\"speedup_total\":%.2f,\"gflops\":%.2f}\n",
           N, blockSize, cpu_ms, kernel_ms, total_gpu_ms, speedup_kern, speedup_total, gflops_kern);

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
    printf(" CUDA Task 1: Matrix Multiplication Using Global Memory\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("====================================================================\n");

    int test_sizes[] = {256, 512, 1024};
    int block_sizes[] = {8, 16, 32};

    if (argc >= 3) {
        int custom_n = atoi(argv[1]);
        int custom_b = atoi(argv[2]);
        benchmarkGlobalMatMul(custom_n, custom_b);
        return 0;
    }

    for (int s = 0; s < 3; ++s) {
        for (int b = 0; b < 3; ++b) {
            benchmarkGlobalMatMul(test_sizes[s], block_sizes[b]);
        }
    }

    return 0;
}
