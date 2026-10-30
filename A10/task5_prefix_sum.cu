/*
 * ==============================================================================
 * HPC Experiment 10 - Task 5: Parallel Prefix Sum / Scan
 * ==============================================================================

 * Description:
 *   Implements inclusive prefix sum (scan):
 *     Input:  [ 3,  1,  7,  0,  4,  2]
 *     Output: [ 3,  4, 11, 11, 15, 17]
 *   Step 1: Sequential CPU implementation.
 *   Step 2: CUDA Shared Memory Parallel Scan using Hillis-Steele tree algorithm.
 *   Step 3: Multi-Block Scan Extension for arbitrary array sizes:
 *           Phase 1: Block-wise scan & extract block sums.
 *           Phase 2: Scan of block sums.
 *           Phase 3: Add scanned block offsets to all elements.
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

#define SCAN_BLOCK_SIZE 256

/* Sequential CPU Inclusive Prefix Sum */
void prefixSumCPU(const int *in, int *out, int N) {
    if (N <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < N; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

/* CUDA Single-Block Hillis-Steele Scan Kernel */
__global__ void hillisSteeleScanKernel(const int *in, int *out, int N) {
    __shared__ int temp[2][SCAN_BLOCK_SIZE];
    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    // Load into shared memory
    if (tid < N) {
        temp[pout][tid] = in[tid];
    } else {
        temp[pout][tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK_SIZE; offset <<= 1) {
        pout = 1 - pout; // swap ping-pong buffers
        pin  = 1 - pout;

        if (tid >= offset) {
            temp[pout][tid] = temp[pin][tid] + temp[pin][tid - offset];
        } else {
            temp[pout][tid] = temp[pin][tid];
        }
        __syncthreads();
    }

    if (tid < N) {
        out[tid] = temp[pout][tid];
    }
}

/* Multi-Block Scan Phase 1: Local Scan within block and write block sums */
__global__ void blockScanKernel(const int *in, int *out, int *blockSums, int N) {
    __shared__ int temp[2][SCAN_BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pout = 0, pin = 1;

    if (gid < N) {
        temp[pout][tid] = in[gid];
    } else {
        temp[pout][tid] = 0;
    }
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK_SIZE; offset <<= 1) {
        pout = 1 - pout;
        pin  = 1 - pout;
        if (tid >= offset) {
            temp[pout][tid] = temp[pin][tid] + temp[pin][tid - offset];
        } else {
            temp[pout][tid] = temp[pin][tid];
        }
        __syncthreads();
    }

    if (gid < N) {
        out[gid] = temp[pout][tid];
    }

    // Save block total sum
    if (blockSums != NULL && tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = temp[pout][tid];
    }
}

/* Multi-Block Scan Phase 3: Add block offset to all elements in block */
__global__ void addBlockOffsetsKernel(int *out, const int *scannedBlockSums, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) {
        out[gid] += scannedBlockSums[blockIdx.x - 1];
    }
}

/* Multi-Block Scan Host Orchestrator */
void multiBlockScan(const int *d_in, int *d_out, int N) {
    int numBlocks = (N + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
    if (numBlocks <= 1) {
        hillisSteeleScanKernel<<<1, SCAN_BLOCK_SIZE>>>(d_in, d_out, N);
        return;
    }

    int *d_blockSums, *d_scannedBlockSums;
    CHECK_CUDA(cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_scannedBlockSums, numBlocks * sizeof(int)));

    // Phase 1: Local scan per block
    blockScanKernel<<<numBlocks, SCAN_BLOCK_SIZE>>>(d_in, d_out, d_blockSums, N);

    // Phase 2: Recursively / directly scan block sums
    if (numBlocks <= SCAN_BLOCK_SIZE) {
        hillisSteeleScanKernel<<<1, SCAN_BLOCK_SIZE>>>(d_blockSums, d_scannedBlockSums, numBlocks);
    } else {
        multiBlockScan(d_blockSums, d_scannedBlockSums, numBlocks);
    }

    // Phase 3: Add scanned offsets
    addBlockOffsetsKernel<<<numBlocks, SCAN_BLOCK_SIZE>>>(d_out, d_scannedBlockSums, N);

    CHECK_CUDA(cudaFree(d_blockSums));
    CHECK_CUDA(cudaFree(d_scannedBlockSums));
}

void testSmallExample() {
    printf("--- Running Example from PDF ---\n");
    int input[] = {3, 1, 7, 0, 4, 2};
    int N = 6;
    int cpu_out[6] = {0};
    int gpu_out[6] = {0};

    prefixSumCPU(input, cpu_out, N);

    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_in, input, N * sizeof(int), cudaMemcpyHostToDevice));

    hillisSteeleScanKernel<<<1, SCAN_BLOCK_SIZE>>>(d_in, d_out, N);
    CHECK_CUDA(cudaMemcpy(gpu_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Input:      [ ");
    for (int i = 0; i < N; ++i) printf("%d ", input[i]);
    printf("]\nCPU Output: [ ");
    for (int i = 0; i < N; ++i) printf("%d ", cpu_out[i]);
    printf("]\nGPU Output: [ ");
    for (int i = 0; i < N; ++i) printf("%d ", gpu_out[i]);
    printf("]\n\n");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

void benchmarkPrefixSum(int N) {
    printf("======================================================\n");
    printf(" Prefix Sum Benchmark: N = %d elements\n", N);
    printf("======================================================\n");

    size_t bytes = (size_t)N * sizeof(int);
    int *h_in = (int*)malloc(bytes);
    int *h_out_cpu = (int*)malloc(bytes);
    int *h_out_gpu = (int*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_in[i] = (i % 5) + 1;
    }

    /* CPU Timing */
    auto t0 = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_in, h_out_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* GPU Execution */
    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    multiBlockScan(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    multiBlockScan(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    /* Verify */
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (h_out_cpu[i] != h_out_gpu[i]) {
            printf("  Mismatch at index %d: CPU=%d, GPU=%d\n", i, h_out_cpu[i], h_out_gpu[i]);
            match = false;
            break;
        }
    }
    printf("  Verification : %s\n", match ? "PASSED" : "FAILED");
    printf("  CPU Time     : %10.3f ms\n", cpu_ms);
    printf("  CUDA Scan    : %10.3f ms\n", gpu_ms);
    printf("  Speedup      : %10.2fx\n", cpu_ms / gpu_ms);

    printf("RESULT_JSON:{\"type\":\"prefix_sum\",\"size\":%d,\"cpu_ms\":%.4f,\"gpu_ms\":%.4f,\"speedup\":%.2f}\n",
           N, cpu_ms, gpu_ms, cpu_ms / gpu_ms);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
}

int main(int argc, char **argv) {
    printf("====================================================================\n");
    printf(" CUDA Task 5: Parallel Prefix Sum / Scan\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("====================================================================\n\n");

    testSmallExample();

    int sizes[] = {100000, 1000000, 10000000};
    if (argc >= 2) {
        int custom_n = atoi(argv[1]);
        benchmarkPrefixSum(custom_n);
        return 0;
    }

    for (int i = 0; i < 3; ++i) {
        benchmarkPrefixSum(sizes[i]);
    }

    return 0;
}
