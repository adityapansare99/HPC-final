/*
 * ==============================================================================
 * HPC Experiment 9 - Task 5: CUDA Memory Management
 * ==============================================================================

 * Description:
 *   Implements vector addition using CUDA device memory with explicit
 *   management of: cudaMalloc(), cudaMemcpy(), cudaFree().
 *   Demonstrates the data movement pipeline:
 *     Host Memory → cudaMemcpy(H2D) → Device Memory → Kernel → 
 *     Device Result → cudaMemcpy(D2H) → Host Result
 *   Separately times each phase and verifies correctness.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ---- CUDA Kernel: Vector Addition ---- */
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

/* ---- Sequential CPU Version ---- */
void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    int N = 1000000;
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    size_t size = N * sizeof(float);
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    /* Step 1: Allocate Host Memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return 1;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i * 0.0005f) * 50.0f;
        h_B[i] = cosf(i * 0.0005f) * 50.0f;
    }

    /* CPU sequential computation for verification */
    vectorAddCPU(h_A, h_B, h_C_cpu, N);

    /* Step 2: Allocate Device Memory using cudaMalloc() */
    float *d_A, *d_B, *d_C;

    cudaEvent_t ev_alloc_start, ev_alloc_end;
    cudaEventCreate(&ev_alloc_start);
    cudaEventCreate(&ev_alloc_end);

    cudaEventRecord(ev_alloc_start);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaEventRecord(ev_alloc_end);
    cudaEventSynchronize(ev_alloc_end);

    float alloc_ms = 0;
    cudaEventElapsedTime(&alloc_ms, ev_alloc_start, ev_alloc_end);

    /* Step 3: Host-to-Device Transfer using cudaMemcpy() */
    cudaEvent_t ev_h2d_start, ev_h2d_end;
    cudaEventCreate(&ev_h2d_start);
    cudaEventCreate(&ev_h2d_end);

    cudaEventRecord(ev_h2d_start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d_end);
    cudaEventSynchronize(ev_h2d_end);

    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_end);

    /* Step 4: Kernel Execution on Device */
    cudaEvent_t ev_kernel_start, ev_kernel_end;
    cudaEventCreate(&ev_kernel_start);
    cudaEventCreate(&ev_kernel_end);

    cudaEventRecord(ev_kernel_start);
    vectorAddKernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(ev_kernel_end);
    cudaEventSynchronize(ev_kernel_end);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_end);

    /* Step 5: Device-to-Host Transfer using cudaMemcpy() */
    cudaEvent_t ev_d2h_start, ev_d2h_end;
    cudaEventCreate(&ev_d2h_start);
    cudaEventCreate(&ev_d2h_end);

    cudaEventRecord(ev_d2h_start);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h_end);
    cudaEventSynchronize(ev_d2h_end);

    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_end);

    /* Step 6: Correctness Verification */
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) mismatches++;
    }

    float total_gpu_ms = alloc_ms + h2d_ms + kernel_ms + d2h_ms;

    if (benchmark_mode) {
        printf("DATA_MEMMAN|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%d|%.6e\n",
               N, alloc_ms, h2d_ms, kernel_ms, d2h_ms, total_gpu_ms,
               mismatches, max_diff);
    } else {
        printf("====================================================================\n");
        printf("  Task 5: CUDA Memory Management — Detailed Phase Timing\n");
        printf("====================================================================\n");
        printf("  Vector Size (N)       : %d\n", N);
        printf("  Data Size             : %.2f MB\n", (3.0 * size) / (1024 * 1024));
        printf("--------------------------------------------------------------------\n");
        printf("  Phase Breakdown:\n");
        printf("    cudaMalloc()        : %.4f ms  [Device memory allocation]\n", alloc_ms);
        printf("    cudaMemcpy(H→D)     : %.4f ms  [Host → Device transfer]\n", h2d_ms);
        printf("    Kernel Execution    : %.4f ms  [GPU computation]\n", kernel_ms);
        printf("    cudaMemcpy(D→H)     : %.4f ms  [Device → Host transfer]\n", d2h_ms);
        printf("    TOTAL GPU Pipeline  : %.4f ms\n", total_gpu_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  Transfer Overhead     : %.2f%% of total GPU time\n",
               (h2d_ms + d2h_ms) / total_gpu_ms * 100.0);
        printf("  Computation Fraction  : %.2f%% of total GPU time\n",
               kernel_ms / total_gpu_ms * 100.0);
        printf("--------------------------------------------------------------------\n");
        printf("  Correctness Check     : %s\n",
               mismatches == 0 ? "PASSED (all elements match)" : "FAILED");
        printf("  Max Absolute Diff     : %.6e\n", max_diff);
        printf("  Mismatches            : %d / %d\n", mismatches, N);
        printf("====================================================================\n");
    }

    /* Step 7: Free Device Memory using cudaFree() */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(ev_alloc_start);
    cudaEventDestroy(ev_alloc_end);
    cudaEventDestroy(ev_h2d_start);
    cudaEventDestroy(ev_h2d_end);
    cudaEventDestroy(ev_kernel_start);
    cudaEventDestroy(ev_kernel_end);
    cudaEventDestroy(ev_d2h_start);
    cudaEventDestroy(ev_d2h_end);

    /* Free Host Memory */
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
