/*
 * ==============================================================================
 * HPC Experiment 10 - Challenge 1: Optimized Parallel Reduction
 * ==============================================================================

 *
 * Description:
 *   Progressive multi-stage optimization of Parallel Reduction:
 *     Kernel 1: Naive Interleaved Addressing (Severe warp divergence & bank conflicts)
 *     Kernel 2: Shared Memory Reduction with Sequential Addressing (Eliminates divergence)
 *     Kernel 3: Highly Optimized Reduction with Warp Unrolling & Shuffle Intrinsics (__shfl_down_sync)
 *               (Zero shared memory bank conflict, eliminates __syncthreads in last warp)
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

/* --- Kernel 1: Naive Interleaved Addressing --- */
__global__ void reduceNaiveInterleaved(const float *g_idata, float *g_odata, int n) {
    __shared__ float sdata[BLOCK_DIM];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Stride doubles each iteration: causes extreme branch divergence in warps
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

/* --- Kernel 2: Sequential Addressing (No Divergence) --- */
__global__ void reduceSequentialAddressing(const float *g_idata, float *g_odata, int n) {
    __shared__ float sdata[BLOCK_DIM];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

/* --- Kernel 3: Fully Optimized with Warp Shuffle (__shfl_down_sync) --- */
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarpOptimized(const float *g_idata, float *g_odata, int n) {
    __shared__ float sdata[32]; // Shared memory for 32 warps (up to 1024 threads)
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int wid = tid / 32;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Grid-stride / dual load
    float sum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    // Intra-warp shuffle reduction
    sum = warpReduceSum(sum);

    if (lane == 0) {
        sdata[wid] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    sum = (tid < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
    if (wid == 0) {
        sum = warpReduceSum(sum);
        if (tid == 0) {
            atomicAdd(g_odata, sum);
        }
    }
}

int main() {
    printf("========================================================================================\n");
    printf(" HPC Experiment 10 - Challenge 1: Optimized Reduction (Interleaved vs Seq vs Warp-Shuffle)\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("========================================================================================\n\n");

    int N = 20000000; // 20M elements
    size_t bytes = (size_t)N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_out, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float zero = 0.0f;
    float time1 = 0.0f, time2 = 0.0f, time3 = 0.0f;

    // Kernel 1: Naive Interleaved
    CHECK_CUDA(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int grid1 = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    CHECK_CUDA(cudaEventRecord(start));
    reduceNaiveInterleaved<<<grid1, BLOCK_DIM>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time1, start, stop));

    // Kernel 2: Sequential Addressing
    CHECK_CUDA(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
    int grid2 = (N + (BLOCK_DIM * 2) - 1) / (BLOCK_DIM * 2);
    CHECK_CUDA(cudaEventRecord(start));
    reduceSequentialAddressing<<<grid2, BLOCK_DIM>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time2, start, stop));

    // Kernel 3: Warp Shuffle
    CHECK_CUDA(cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(start));
    reduceWarpOptimized<<<grid2, BLOCK_DIM>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time3, start, stop));

    float final_res = 0.0f;
    CHECK_CUDA(cudaMemcpy(&final_res, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Array Size: %d elements (%.1f MB)\n", N, bytes / (1024.0 * 1024.0));
    printf("Verified Sum: %.1f (Expected: %.1f)\n\n", final_res, (float)N);
    printf("+-----------------------------------+---------------+-----------------------+\n");
    printf("| Optimization Stage                | Execution Time| Speedup vs Naive      |\n");
    printf("+-----------------------------------+---------------+-----------------------+\n");
    printf("| 1. Naive Interleaved Addressing   |   %8.3f ms |        1.00x          |\n", time1);
    printf("| 2. Shared Sequential Addressing   |   %8.3f ms |        %5.2fx          |\n", time2, time1 / time2);
    printf("| 3. Warp Shuffle (__shfl_down)     |   %8.3f ms |        %5.2fx          |\n", time3, time1 / time3);
    printf("+-----------------------------------+---------------+-----------------------+\n\n");

    printf("RESULT_JSON:{\"type\":\"challenge1_reduction\",\"size\":%d,\"naive_ms\":%.4f,\"sequential_ms\":%.4f,\"warp_shuffle_ms\":%.4f,\"speedup_seq\":%.2f,\"speedup_warp\":%.2f}\n",
           N, time1, time2, time3, time1 / time2, time1 / time3);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_in);

    return 0;
}
