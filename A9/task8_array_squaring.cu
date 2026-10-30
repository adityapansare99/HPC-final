/*
 * ==============================================================================
 * HPC Experiment 9 - Task 8: 1D Array Processing — Array Squaring (Option C)
 * ==============================================================================

 * Description:
 *   Implements array squaring C[i] = A[i] * A[i] using:
 *     - Sequential CPU version
 *     - CUDA GPU version
 *   Uses global thread ID, device memory, and host-device data transfer.
 *   Compares performance and verifies correctness.
 *
 * Compile: nvcc task8_array_squaring.cu -o task8_array_squaring -lm
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

/* ---- Sequential CPU: Array Squaring ---- */
void arraySquareCPU(const float *A, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * A[i];
    }
}

/* ---- CUDA Kernel: Array Squaring ---- */
__global__ void arraySquareKernel(const float *A, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] * A[id];
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

    /* Host memory */
    float *h_A = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    if (!h_A || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return 1;
    }

    /* Initialize array */
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i * 0.0003f) * 75.0f + 0.5f;
    }

    /* ---- Sequential CPU Version ---- */
    auto ts = std::chrono::high_resolution_clock::now();
    arraySquareCPU(h_A, h_C_cpu, N);
    auto te = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(te - ts).count();

    /* ---- CUDA GPU Version ---- */
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);

    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);

    cudaEventRecord(ev_start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaEventRecord(ev_h2d);

    arraySquareKernel<<<num_blocks, threads_per_block>>>(d_A, d_C, N);
    cudaEventRecord(ev_kernel);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_d2h);
    cudaEventSynchronize(ev_d2h);

    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, ev_start, ev_h2d);
    cudaEventElapsedTime(&kernel_ms, ev_h2d, ev_kernel);
    cudaEventElapsedTime(&d2h_ms, ev_kernel, ev_d2h);
    float gpu_total_ms = h2d_ms + kernel_ms + d2h_ms;

    /* Correctness verification */
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) mismatches++;
    }

    float speedup = cpu_ms / gpu_total_ms;

    if (benchmark_mode) {
        printf("DATA_SQUARE|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%d|%.6e\n",
               N, cpu_ms, h2d_ms, kernel_ms, d2h_ms, gpu_total_ms, speedup,
               mismatches, max_diff);
    } else {
        printf("====================================================================\n");
        printf("  Task 8: 1D Array Processing — Array Squaring (C[i] = A[i]^2)\n");
        printf("====================================================================\n");
        printf("  Array Size (N)        : %d\n", N);
        printf("  Threads per Block     : %d\n", threads_per_block);
        printf("  Number of Blocks      : %d\n", num_blocks);
        printf("--------------------------------------------------------------------\n");
        printf("  CPU Sequential Time   : %.4f ms\n", cpu_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  GPU Host→Device       : %.4f ms\n", h2d_ms);
        printf("  GPU Kernel Execution  : %.4f ms\n", kernel_ms);
        printf("  GPU Device→Host       : %.4f ms\n", d2h_ms);
        printf("  GPU Total Time        : %.4f ms\n", gpu_total_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  Speedup (CPU/GPU)     : %.2fx\n", speedup);
        printf("  Kernel-only Speedup   : %.2fx\n", cpu_ms / kernel_ms);
        printf("--------------------------------------------------------------------\n");
        printf("  Correctness Verification:\n");
        printf("    CPU Result[0]       : %.6f\n", h_C_cpu[0]);
        printf("    GPU Result[0]       : %.6f\n", h_C_gpu[0]);
        printf("    Max Absolute Diff   : %.6e\n", max_diff);
        printf("    Status              : %s (%d mismatches)\n",
               mismatches == 0 ? "PASSED" : "FAILED", mismatches);
        printf("====================================================================\n");
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_h2d);
    cudaEventDestroy(ev_kernel);
    cudaEventDestroy(ev_d2h);
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
