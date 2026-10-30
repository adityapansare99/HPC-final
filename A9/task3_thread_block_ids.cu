/*
 * ==============================================================================
 * HPC Experiment 9 - Task 3: Understand Thread and Block IDs
 * ==============================================================================

 * Description:
 *   Demonstrates CUDA thread hierarchy by displaying:
 *     - Block ID (blockIdx.x)
 *     - Thread ID within block (threadIdx.x)
 *     - Global Thread ID (blockIdx.x * blockDim.x + threadIdx.x)
 *   Run with 2 blocks × 4 threads per block.
 * ==============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void showIdsKernel() {
    int blockId  = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Block %d | Thread %d | Global ID %d\n",
           blockId, threadId, globalId);
}

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    if (argc > 1) benchmark_mode = atoi(argv[1]);

    int blocks  = 2;
    int threads = 4;

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 3: Thread and Block IDs\n");
        printf("  Configuration: %d blocks × %d threads/block = %d total threads\n",
               blocks, threads, blocks * threads);
        printf("====================================================================\n");
        printf("  Expected Global IDs: 0 through %d\n\n", blocks * threads - 1);
        fflush(stdout);
    }

    showIdsKernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    if (benchmark_mode) {
        printf("DATA_IDS|%d|%d|%d\n", blocks, threads, blocks * threads);
    }

    if (!benchmark_mode) {
        printf("\n====================================================================\n");
        printf("  Global ID = blockIdx.x * blockDim.x + threadIdx.x\n");
        printf("  = blockIdx.x * %d + threadIdx.x\n", threads);
        printf("====================================================================\n");
    }

    return 0;
}
