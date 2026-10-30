/*
 * ==============================================================================
 * HPC Experiment 9 - Task 2: First CUDA Program — Hello from GPU
 * ==============================================================================

 * Description:
 *   Launches a CUDA kernel where every thread prints a greeting message.
 *   Demonstrates thread indexing with different block/thread configurations:
 *     - 1 block, 8 threads
 *     - 1 block, 16 threads
 *     - 2 blocks, 8 threads per block
 *     - 4 blocks, 8 threads per block
 * ==============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU Thread %d (Block %d, Thread %d)\n",
           globalId, blockIdx.x, threadIdx.x);
}

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    if (argc > 1) benchmark_mode = atoi(argv[1]);

    /* Configuration table: (blocks, threads_per_block) */
    int configs[][2] = {
        {1,  8},
        {1, 16},
        {2,  8},
        {4,  8}
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    for (int c = 0; c < num_configs; c++) {
        int blocks = configs[c][0];
        int threads = configs[c][1];
        int total = blocks * threads;

        if (!benchmark_mode) {
            printf("====================================================================\n");
            printf("  Configuration: %d block(s), %d threads/block = %d total threads\n",
                   blocks, threads, total);
            printf("====================================================================\n");
            fflush(stdout);
        }

        helloKernel<<<blocks, threads>>>();
        cudaDeviceSynchronize();

        if (benchmark_mode) {
            printf("DATA_HELLO|%d|%d|%d\n", blocks, threads, total);
        }

        if (!benchmark_mode) {
            printf("\n");
        }
    }

    return 0;
}
