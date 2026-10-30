/*
 * ==============================================================================
 * HPC Experiment 10 - Challenge 2: Multi-Block Prefix Sum (Arbitrary Size Scan)
 * ==============================================================================

 * Description:
 *   Extends prefix sum to arrays larger than a single CUDA thread block.
 *   Three-Phase Algorithm:
 *     Phase 1: Block-wise Scan (Hillis-Steele per block) & Block Sums Extraction
 *     Phase 2: Inclusive Scan of Block Sums array (recursive or single-block)
 *     Phase 3: Element-wise Offset Addition (Add scanned block sum (b-1) to block b)
 *   Demonstrates true multi-block scalability on 1M, 5M, and 20M elements.
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

#define SCAN_BLOCK 512

/* Sequential CPU Reference */
void cpuScan(const int *in, int *out, int N) {
    if (N <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < N; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

/* Phase 1: Local Scan and Extract Block Sum */
__global__ void phase1_BlockScan(const int *in, int *out, int *blockSums, int N) {
    __shared__ int temp[2][SCAN_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int pout = 0, pin = 1;

    temp[pout][tid] = (gid < N) ? in[gid] : 0;
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
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

    if (blockSums != NULL && tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = temp[pout][tid];
    }
}

/* Phase 2: Single-block Scan of Block Sums (when numBlocks <= SCAN_BLOCK) */
__global__ void phase2_ScanBlockSums(const int *in, int *out, int numBlocks) {
    __shared__ int temp[2][SCAN_BLOCK];
    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    temp[pout][tid] = (tid < numBlocks) ? in[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
        pout = 1 - pout;
        pin  = 1 - pout;
        if (tid >= offset) {
            temp[pout][tid] = temp[pin][tid] + temp[pin][tid - offset];
        } else {
            temp[pout][tid] = temp[pin][tid];
        }
        __syncthreads();
    }

    if (tid < numBlocks) {
        out[tid] = temp[pout][tid];
    }
}

/* Phase 3: Add Scanned Offsets */
__global__ void phase3_AddOffsets(int *out, const int *scannedBlockSums, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) {
        out[gid] += scannedBlockSums[blockIdx.x - 1];
    }
}

void multiBlockPrefixSum(const int *d_in, int *d_out, int N) {
    int numBlocks = (N + SCAN_BLOCK - 1) / SCAN_BLOCK;
    if (numBlocks <= 1) {
        phase1_BlockScan<<<1, SCAN_BLOCK>>>(d_in, d_out, NULL, N);
        return;
    }

    int *d_blockSums, *d_scannedBlockSums;
    CHECK_CUDA(cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_scannedBlockSums, numBlocks * sizeof(int)));

    // Phase 1: Local scan
    phase1_BlockScan<<<numBlocks, SCAN_BLOCK>>>(d_in, d_out, d_blockSums, N);

    // Phase 2: Scan block sums
    if (numBlocks <= SCAN_BLOCK) {
        phase2_ScanBlockSums<<<1, SCAN_BLOCK>>>(d_blockSums, d_scannedBlockSums, numBlocks);
    } else {
        // Recursive multi-block scan
        multiBlockPrefixSum(d_blockSums, d_scannedBlockSums, numBlocks);
    }

    // Phase 3: Add offsets
    phase3_AddOffsets<<<numBlocks, SCAN_BLOCK>>>(d_out, d_scannedBlockSums, N);

    CHECK_CUDA(cudaFree(d_blockSums));
    CHECK_CUDA(cudaFree(d_scannedBlockSums));
}

int main() {
    printf("========================================================================================\n");
    printf(" HPC Experiment 10 - Challenge 2: Scalable Multi-Block Prefix Sum\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("========================================================================================\n\n");

    int test_sizes[] = {1000000, 5000000, 20000000};

    for (int s = 0; s < 3; ++s) {
        int N = test_sizes[s];
        size_t bytes = (size_t)N * sizeof(int);
        int *h_in = (int*)malloc(bytes);
        int *h_out_cpu = (int*)malloc(bytes);
        int *h_out_gpu = (int*)malloc(bytes);

        for (int i = 0; i < N; ++i) h_in[i] = (i % 3) + 1;

        auto t0 = std::chrono::high_resolution_clock::now();
        cpuScan(h_in, h_out_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int *d_in, *d_out;
        CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Warm-up
        multiBlockPrefixSum(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        multiBlockPrefixSum(d_in, d_out, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float gpu_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));
        CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (h_out_cpu[i] != h_out_gpu[i]) {
                printf("  Mismatch at index %d: CPU=%d, GPU=%d\n", i, h_out_cpu[i], h_out_gpu[i]);
                ok = false;
                break;
            }
        }

        printf("  Array Size: %8d | Verification: %s | CPU: %8.2f ms | CUDA Multi-Block: %6.2f ms | Speedup: %5.2fx\n",
               N, ok ? "PASSED" : "FAILED", cpu_ms, gpu_ms, cpu_ms / gpu_ms);

        printf("RESULT_JSON:{\"type\":\"challenge2_scan\",\"size\":%d,\"cpu_ms\":%.4f,\"gpu_ms\":%.4f,\"speedup\":%.2f}\n",
               N, cpu_ms, gpu_ms, cpu_ms / gpu_ms);

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        free(h_in);
        free(h_out_cpu);
        free(h_out_gpu);
    }

    return 0;
}
