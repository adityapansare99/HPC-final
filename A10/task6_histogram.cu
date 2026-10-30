/*
 * ==============================================================================
 * HPC Experiment 10 - Task 6: CUDA Histogram & Atomic Contention Analysis
 * ==============================================================================

 * Description:
 *   Calculates the frequency of occurrence of discrete values (bins):
 *     Input:  [ 1, 2, 1, 3, 2, 1, 4, 3, 2 ]
 *     Bins:   0:0, 1:3, 2:3, 3:2, 4:1
 *   Version A: Sequential CPU Histogram.
 *   Version B: CUDA Global Memory Atomic Histogram:
 *              atomicAdd(&d_hist[val], 1);
 *   Version C: CUDA Shared Memory Privatized Histogram:
 *              Each block initializes __shared__ int local_hist[NUM_BINS] to 0,
 *              performs fast on-chip shared atomicAdd, synchronizes, then
 *              atomically accumulates local bins into the global histogram.
 *   Evaluates both:
 *     1. Uniform Distribution (scattered bin updates, low contention).
 *     2. Concentrated / Skewed Distribution (heavy updates to few bins, severe contention).
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

#define HIST_BLOCK_SIZE 256
#define NUM_BINS 256

/* Sequential CPU Histogram */
void histogramCPU(const int *data, int N, int *hist, int numBins) {
    for (int b = 0; b < numBins; ++b) hist[b] = 0;
    for (int i = 0; i < N; ++i) {
        int v = data[i];
        if (v >= 0 && v < numBins) {
            hist[v]++;
        }
    }
}

/* Version B: CUDA Global Memory Atomic Histogram */
__global__ void histGlobalAtomicKernel(const int *data, int N, int *hist, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        int val = data[i];
        if (val >= 0 && val < numBins) {
            atomicAdd(&hist[val], 1);
        }
    }
}

/* Version C: CUDA Shared Memory Privatized Histogram */
__global__ void histSharedPrivatizedKernel(const int *data, int N, int *hist, int numBins) {
    __shared__ int local_hist[NUM_BINS];

    // Initialize shared memory sub-histogram to zero
    int tid = threadIdx.x;
    for (int b = tid; b < numBins; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    // Accumulate locally into fast on-chip shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        int val = data[i];
        if (val >= 0 && val < numBins) {
            atomicAdd(&local_hist[val], 1);
        }
    }
    __syncthreads();

    // Cooperatively commit block-level sub-histogram to global memory
    for (int b = tid; b < numBins; b += blockDim.x) {
        if (local_hist[b] > 0) {
            atomicAdd(&hist[b], local_hist[b]);
        }
    }
}

void testPDFExample() {
    printf("--- Running Example from PDF ---\n");
    int input[] = {1, 2, 1, 3, 2, 1, 4, 3, 2};
    int N = 9;
    int numBins = 5;
    int cpu_hist[5] = {0};
    int gpu_hist[5] = {0};

    histogramCPU(input, N, cpu_hist, numBins);

    int *d_data, *d_hist;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_hist, numBins * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, input, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hist, 0, numBins * sizeof(int)));

    histGlobalAtomicKernel<<<1, HIST_BLOCK_SIZE>>>(d_data, N, d_hist, numBins);
    CHECK_CUDA(cudaMemcpy(gpu_hist, d_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Input: [ 1, 2, 1, 3, 2, 1, 4, 3, 2 ]\n");
    printf("Value | CPU Count | GPU Count\n");
    printf("------+-----------+----------\n");
    for (int b = 0; b < numBins; ++b) {
        printf("  %d   |     %d     |     %d\n", b, cpu_hist[b], gpu_hist[b]);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));
}

void benchmarkHistogram(int N, bool skewed) {
    const char *dist_type = skewed ? "Concentrated / Skewed (Heavy Contention)" : "Uniform Distribution";
    printf("====================================================================\n");
    printf(" Histogram Benchmark: N = %d items | Distribution: %s\n", N, dist_type);
    printf("====================================================================\n");

    size_t bytes = (size_t)N * sizeof(int);
    int *h_data = (int*)malloc(bytes);
    int *h_cpu_hist = (int*)malloc(NUM_BINS * sizeof(int));
    int *h_glob_hist = (int*)malloc(NUM_BINS * sizeof(int));
    int *h_sh_hist = (int*)malloc(NUM_BINS * sizeof(int));

    srand(42);
    for (int i = 0; i < N; ++i) {
        if (skewed) {
            // 90% of all elements hit bin 0 or 1
            if ((rand() % 100) < 90) {
                h_data[i] = rand() % 2;
            } else {
                h_data[i] = rand() % NUM_BINS;
            }
        } else {
            // Uniformly distributed across all 256 bins
            h_data[i] = rand() % NUM_BINS;
        }
    }

    /* 1. CPU Histogram */
    auto t0 = std::chrono::high_resolution_clock::now();
    histogramCPU(h_data, N, h_cpu_hist, NUM_BINS);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* 2. GPU Buffers */
    int *d_data, *d_hist;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (N + HIST_BLOCK_SIZE - 1) / HIST_BLOCK_SIZE;
    if (numBlocks > 1024) numBlocks = 1024; // limit grid size for persistent loop

    cudaEvent_t startG, stopG, startS, stopS;
    CHECK_CUDA(cudaEventCreate(&startG));
    CHECK_CUDA(cudaEventCreate(&stopG));
    CHECK_CUDA(cudaEventCreate(&startS));
    CHECK_CUDA(cudaEventCreate(&stopS));

    /* --- Version B: Global Atomics --- */
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(startG));
    histGlobalAtomicKernel<<<numBlocks, HIST_BLOCK_SIZE>>>(d_data, N, d_hist, NUM_BINS);
    CHECK_CUDA(cudaEventRecord(stopG));
    CHECK_CUDA(cudaEventSynchronize(stopG));
    float glob_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&glob_ms, startG, stopG));
    CHECK_CUDA(cudaMemcpy(h_glob_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    /* --- Version C: Shared Memory Privatized --- */
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaEventRecord(startS));
    histSharedPrivatizedKernel<<<numBlocks, HIST_BLOCK_SIZE>>>(d_data, N, d_hist, NUM_BINS);
    CHECK_CUDA(cudaEventRecord(stopS));
    CHECK_CUDA(cudaEventSynchronize(stopS));
    float shared_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, startS, stopS));
    CHECK_CUDA(cudaMemcpy(h_sh_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    /* Verification */
    bool match = true;
    for (int b = 0; b < NUM_BINS; ++b) {
        if (h_cpu_hist[b] != h_sh_hist[b] || h_cpu_hist[b] != h_glob_hist[b]) {
            printf("  Mismatch at bin %d: CPU=%d, Glob=%d, Shared=%d\n",
                   b, h_cpu_hist[b], h_glob_hist[b], h_sh_hist[b]);
            match = false;
            break;
        }
    }

    printf("  Verification            : %s\n", match ? "PASSED" : "FAILED");
    printf("  CPU Execution Time      : %10.3f ms\n", cpu_ms);
    printf("  Global Atomic Time      : %10.3f ms\n", glob_ms);
    printf("  Shared Privatized Time  : %10.3f ms\n", shared_ms);
    printf("  Speedup (Shared vs Glob): %10.2fx\n", glob_ms / shared_ms);
    printf("  Speedup (Shared vs CPU) : %10.2fx\n\n", cpu_ms / shared_ms);

    printf("RESULT_JSON:{\"type\":\"histogram\",\"size\":%d,\"skewed\":%s,\"cpu_ms\":%.4f,\"global_ms\":%.4f,\"shared_ms\":%.4f,\"speedup_vs_global\":%.2f,\"speedup_vs_cpu\":%.2f}\n",
           N, skewed ? "true" : "false", cpu_ms, glob_ms, shared_ms, glob_ms / shared_ms, cpu_ms / shared_ms);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_hist));
    CHECK_CUDA(cudaEventDestroy(startG));
    CHECK_CUDA(cudaEventDestroy(stopG));
    CHECK_CUDA(cudaEventDestroy(startS));
    CHECK_CUDA(cudaEventDestroy(stopS));
    free(h_data);
    free(h_cpu_hist);
    free(h_glob_hist);
    free(h_sh_hist);
}

int main(int argc, char **argv) {
    printf("====================================================================\n");
    printf(" CUDA Task 6: CUDA Histogram (Global vs Shared Memory Atomics)\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("====================================================================\n\n");

    testPDFExample();

    int test_sizes[] = {1000000, 10000000, 50000000};
    for (int i = 0; i < 3; ++i) {
        benchmarkHistogram(test_sizes[i], false); // uniform
        benchmarkHistogram(test_sizes[i], true);  // skewed / high contention
    }

    return 0;
}
