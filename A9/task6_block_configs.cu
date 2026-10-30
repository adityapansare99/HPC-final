/*
 * ==============================================================================
 * HPC Experiment 9 - Task 6: Vector Addition with Different Block Configurations
 * ==============================================================================

 * Description:
 *   Runs vector addition with a fixed vector size (N = 1,000,000) using
 *   different block sizes: 64, 128, 256, 512 threads/block.
 *   Calculates the number of blocks required for each configuration.
 *   Analyzes the impact of block size on kernel execution time.
 *   Includes boundary check (id < N) to handle non-exact multiples.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
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

    /* Host allocations */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);
    float *h_C_gpu = (float *)malloc(size);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return 1;
    }

    /* Initialize */
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i * 0.001f) * 100.0f;
        h_B[i] = cosf(i * 0.001f) * 100.0f;
    }

    /* CPU sequential baseline */
    auto t_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto t_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    /* Device allocations */
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    /* Copy to device once */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    /* Block size configurations */
    int block_sizes[] = {64, 128, 256, 512};
    int num_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 6: Vector Addition — Block Size Comparison\n");
        printf("  Vector Size (N) = %d | CPU Time = %.4f ms\n", N, cpu_time_ms);
        printf("====================================================================\n");
        printf("  %-14s %-10s %-14s %-14s %-10s\n",
               "Threads/Block", "Blocks", "Kernel (ms)", "Total (ms)", "Speedup");
        printf("  %-14s %-10s %-14s %-14s %-10s\n",
               "-------------", "------", "-----------", "----------", "-------");
    }

    for (int c = 0; c < num_configs; c++) {
        int tpb = block_sizes[c];
        int num_blocks = (N + tpb - 1) / tpb;

        /* Warm-up run */
        vectorAddKernel<<<num_blocks, tpb>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        /* Timed run (average of 5 iterations) */
        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

        int runs = 5;
        float total_kernel_ms = 0.0f;

        for (int r = 0; r < runs; r++) {
            cudaEventRecord(ev_start);
            vectorAddKernel<<<num_blocks, tpb>>>(d_A, d_B, d_C, N);
            cudaEventRecord(ev_end);
            cudaEventSynchronize(ev_end);
            float elapsed = 0;
            cudaEventElapsedTime(&elapsed, ev_start, ev_end);
            total_kernel_ms += elapsed;
        }

        float avg_kernel_ms = total_kernel_ms / runs;

        /* Include H2D and D2H for total time */
        cudaEvent_t ev_total_start, ev_total_end;
        cudaEventCreate(&ev_total_start);
        cudaEventCreate(&ev_total_end);

        cudaEventRecord(ev_total_start);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        vectorAddKernel<<<num_blocks, tpb>>>(d_A, d_B, d_C, N);
        cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_total_end);
        cudaEventSynchronize(ev_total_end);

        float total_ms = 0;
        cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_end);

        /* Verify correctness */
        int mismatches = 0;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_C_cpu[i] - h_C_gpu[i]) > 1e-4f) mismatches++;
        }

        float speedup = cpu_time_ms / total_ms;

        if (benchmark_mode) {
            printf("DATA_BLOCKSIZE|%d|%d|%d|%.4f|%.4f|%.4f|%d\n",
                   N, tpb, num_blocks, avg_kernel_ms, total_ms, (float)cpu_time_ms, mismatches);
        } else {
            printf("  %-14d %-10d %-14.4f %-14.4f %-10.2f  %s\n",
                   tpb, num_blocks, avg_kernel_ms, total_ms, speedup,
                   mismatches == 0 ? "✓" : "✗");
        }

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
        cudaEventDestroy(ev_total_start);
        cudaEventDestroy(ev_total_end);
    }

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Note: Blocks = ceil(N / threads_per_block)\n");
        printf("  Boundary check (id < N) handles N not divisible by block size.\n");
        printf("====================================================================\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
