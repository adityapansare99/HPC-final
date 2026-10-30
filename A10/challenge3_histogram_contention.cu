/*
 * ==============================================================================
 * HPC Experiment 10 - Challenge 3: Advanced Histogram & Atomic Contention Study
 * ==============================================================================

 * Description:
 *   Investigates hardware memory contention on atomic operations:
 *     1. Evaluates effect of varying bins: 16, 64, 256, 1024 bins.
 *     2. Evaluates two distinct distributions:
 *        - Uniform Distribution (threads randomly touch all bins)
 *        - Concentrated / Pathological Skewed (95% threads hit bin 0)
 *     3. Compares Global Atomic Histogram vs Shared Privatized Histogram.
 *   Shows that under high contention, global atomics serialize catastrophically,
 *   while shared memory privatizes contention to fast on-chip L1/shared memory.
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

#define BLOCK_DIM 256

__global__ void globalAtomicHist(const int *data, int N, int *hist, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        int v = data[i];
        if (v >= 0 && v < numBins) {
            atomicAdd(&hist[v], 1);
        }
    }
}

__global__ void sharedPrivatizedHist(const int *data, int N, int *hist, int numBins) {
    extern __shared__ int local_hist[];
    int tid = threadIdx.x;

    for (int b = tid; b < numBins; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        int v = data[i];
        if (v >= 0 && v < numBins) {
            atomicAdd(&local_hist[v], 1);
        }
    }
    __syncthreads();

    for (int b = tid; b < numBins; b += blockDim.x) {
        if (local_hist[b] > 0) {
            atomicAdd(&hist[b], local_hist[b]);
        }
    }
}

int main() {
    printf("========================================================================================\n");
    printf(" HPC Experiment 10 - Challenge 3: Hardware Contention in Parallel Histograms\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("========================================================================================\n\n");

    int N = 20000000; // 20M integers
    size_t bytes = (size_t)N * sizeof(int);
    int *h_data = (int*)malloc(bytes);

    int bin_configs[] = {16, 64, 256, 1024};

    int *d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));

    printf("+----------+----------------------+--------------------+--------------------+------------+\n");
    printf("| Num Bins | Data Distribution    | Global Atomic Time | Shared Memory Time | Speedup    |\n");
    printf("+----------+----------------------+--------------------+--------------------+------------+\n");

    for (int bc = 0; bc < 4; ++bc) {
        int bins = bin_configs[bc];
        int *d_hist;
        CHECK_CUDA(cudaMalloc((void**)&d_hist, bins * sizeof(int)));

        // Test two distributions: Uniform vs Skewed
        for (int dist = 0; dist < 2; ++dist) {
            bool skewed = (dist == 1);
            const char *dist_name = skewed ? "Skewed (95% Peak)    " : "Uniform Distribution ";

            srand(1234);
            for (int i = 0; i < N; ++i) {
                if (skewed) {
                    h_data[i] = ((rand() % 100) < 95) ? 0 : (rand() % bins);
                } else {
                    h_data[i] = rand() % bins;
                }
            }
            CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

            int grid = 512;
            cudaEvent_t startG, stopG, startS, stopS;
            CHECK_CUDA(cudaEventCreate(&startG));
            CHECK_CUDA(cudaEventCreate(&stopG));
            CHECK_CUDA(cudaEventCreate(&startS));
            CHECK_CUDA(cudaEventCreate(&stopS));

            // Global Atomics
            CHECK_CUDA(cudaMemset(d_hist, 0, bins * sizeof(int)));
            CHECK_CUDA(cudaEventRecord(startG));
            globalAtomicHist<<<grid, BLOCK_DIM>>>(d_data, N, d_hist, bins);
            CHECK_CUDA(cudaEventRecord(stopG));
            CHECK_CUDA(cudaEventSynchronize(stopG));
            float timeG = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&timeG, startG, stopG));

            // Shared Privatized
            size_t sharedBytes = bins * sizeof(int);
            CHECK_CUDA(cudaMemset(d_hist, 0, bins * sizeof(int)));
            CHECK_CUDA(cudaEventRecord(startS));
            sharedPrivatizedHist<<<grid, BLOCK_DIM, sharedBytes>>>(d_data, N, d_hist, bins);
            CHECK_CUDA(cudaEventRecord(stopS));
            CHECK_CUDA(cudaEventSynchronize(stopS));
            float timeS = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&timeS, startS, stopS));

            float speedup = timeG / timeS;
            printf("|   %4d   | %s |     %8.3f ms    |     %8.3f ms    |   %5.2fx   |\n",
                   bins, dist_name, timeG, timeS, speedup);

            printf("RESULT_JSON:{\"type\":\"challenge3_hist\",\"bins\":%d,\"skewed\":%s,\"global_ms\":%.4f,\"shared_ms\":%.4f,\"speedup\":%.2f}\n",
                   bins, skewed ? "true" : "false", timeG, timeS, speedup);

            CHECK_CUDA(cudaEventDestroy(startG));
            CHECK_CUDA(cudaEventDestroy(stopG));
            CHECK_CUDA(cudaEventDestroy(startS));
            CHECK_CUDA(cudaEventDestroy(stopS));
        }
        printf("+----------+----------------------+--------------------+--------------------+------------+\n");
        CHECK_CUDA(cudaFree(d_hist));
    }

    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    return 0;
}
