/*
 * ==============================================================================
 * HPC Experiment 9 - Task 4: Vector Addition Using CUDA
 * ==============================================================================

 * Description:
 *   Implements vector addition C[i] = A[i] + B[i] in two versions:
 *     Step 1: Sequential CPU version with timing.
 *     Step 2: CUDA GPU version with timing.
 *   Includes correctness verification comparing CPU and GPU results.
 *   Uses global thread ID: id = blockIdx.x * blockDim.x + threadIdx.x
 *   with boundary check if (id < N).
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

/* ---- Sequential CPU Vector Addition ---- */
void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

/* ---- CUDA Kernel: Vector Addition ---- */
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

/* ---- Initialize vectors with deterministic values ---- */
void initVectors(float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = sinf(i * 0.001f) * 100.0f;
        B[i] = cosf(i * 0.001f) * 100.0f;
    }
}

/* ---- Verify correctness ---- */
int verifyResults(const float *cpu_C, const float *gpu_C, int N, float tolerance) {
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(cpu_C[i] - gpu_C[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (mismatches < 5)
                printf("  Mismatch at index %d: CPU=%.6f, GPU=%.6f, diff=%.6e\n",
                       i, cpu_C[i], gpu_C[i], diff);
            mismatches++;
        }
    }
    printf("  Max absolute difference: %.6e\n", max_diff);
    return mismatches;
}

int main(int argc, char *argv[]) {
    int N = 1000000;
    int threads_per_block = 256;
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) threads_per_block = atoi(argv[2]);
    if (argc > 3) benchmark_mode = atoi(argv[3]);

    size_t size = N * sizeof(float);

    /* Host memory allocation */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed for N=%d\n", N);
        return 1;
    }

    initVectors(h_A, h_B, N);

    /* ---- Step 1: Sequential CPU Version ---- */
    auto t_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto t_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    /* ---- Step 2: CUDA GPU Version ---- */
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    /* Create CUDA events for precise timing */
    cudaEvent_t ev_start, ev_h2d_done, ev_kernel_done, ev_d2h_done;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d_done);
    cudaEventCreate(&ev_kernel_done);
    cudaEventCreate(&ev_d2h_done);

    /* Time: Host-to-Device transfer */
    cudaEventRecord(ev_start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d_done);

    /* Time: Kernel execution */
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    vectorAddKernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(ev_kernel_done);

    /* Time: Device-to-Host transfer */
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h_done);
    cudaEventSynchronize(ev_d2h_done);

    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d_done);
    cudaEventElapsedTime(&kernel_ms, ev_h2d_done, ev_kernel_done);
    cudaEventElapsedTime(&d2h_ms, ev_kernel_done, ev_d2h_done);
    float gpu_total_ms = h2d_ms + kernel_ms + d2h_ms;

    /* Correctness verification */
    int mismatches = verifyResults(h_C_cpu, h_C_gpu, N, 1e-4f);

    if (benchmark_mode) {
        printf("DATA_VECADD|%d|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%d\n",
               N, threads_per_block, cpu_time_ms, h2d_ms, kernel_ms, d2h_ms,
               gpu_total_ms, mismatches);
    } else {
        printf("====================================================================\n");
        printf("  Task 4: Vector Addition — Sequential vs CUDA\n");
        printf("====================================================================\n");
        printf("  Vector Size (N)       : %d\n", N);
        printf("  Threads per Block     : %d\n", threads_per_block);
        printf("  Number of Blocks      : %d\n", num_blocks);
        printf("--------------------------------------------------------------------\n");
        printf("  CPU Sequential Time   : %.4f ms\n", cpu_time_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  GPU Host→Device Time  : %.4f ms\n", h2d_ms);
        printf("  GPU Kernel Time       : %.4f ms\n", kernel_ms);
        printf("  GPU Device→Host Time  : %.4f ms\n", d2h_ms);
        printf("  GPU Total Time        : %.4f ms\n", gpu_total_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  Speedup (CPU/GPU)     : %.2fx\n", cpu_time_ms / gpu_total_ms);
        printf("  Kernel-only Speedup   : %.2fx\n", cpu_time_ms / kernel_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  Correctness Check     : %s (%d mismatches)\n",
               mismatches == 0 ? "PASSED" : "FAILED", mismatches);
        printf("====================================================================\n");
    }

    /* Cleanup */
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_h2d_done);
    cudaEventDestroy(ev_kernel_done);
    cudaEventDestroy(ev_d2h_done);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
