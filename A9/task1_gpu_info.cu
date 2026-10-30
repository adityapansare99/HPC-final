/*
 * ==============================================================================
 * HPC Experiment 9 - Task 1: Explore the GPU Environment
 * ==============================================================================

 * Description:
 *   Queries and displays comprehensive GPU device properties using the
 *   CUDA Runtime API (cudaGetDeviceProperties). Reports GPU name, memory,
 *   compute capability, SM count, max threads/block, max grid dimensions,
 *   warp size, clock rates, and memory bus width.
 * ==============================================================================
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    if (argc > 1) benchmark_mode = atoi(argv[1]);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable GPU detected.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        if (benchmark_mode) {
            printf("DATA_GPU_INFO|%s|%d.%d|%d|%d|%d|%lu|%d|%d|%d\n",
                   prop.name, prop.major, prop.minor,
                   prop.multiProcessorCount,
                   prop.maxThreadsPerBlock,
                   prop.maxThreadsPerMultiProcessor,
                   prop.totalGlobalMem / (1024 * 1024),
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        } else {
            printf("====================================================================\n");
            printf("  Task 1: GPU Environment Information — Device %d\n", dev);
            printf("====================================================================\n");
            printf("  GPU Name                  : %s\n", prop.name);
            printf("  Compute Capability        : %d.%d\n", prop.major, prop.minor);
            printf("  Total Global Memory       : %lu MB (%.2f GB)\n",
                   prop.totalGlobalMem / (1024 * 1024),
                   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
            printf("  Number of SMs             : %d\n", prop.multiProcessorCount);
            printf("  Max Threads per Block     : %d\n", prop.maxThreadsPerBlock);
            printf("  Max Threads per SM        : %d\n", prop.maxThreadsPerMultiProcessor);
            printf("  Warp Size                 : %d\n", prop.warpSize);
            printf("  Max Block Dimensions      : (%d, %d, %d)\n",
                   prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("  Max Grid Dimensions       : (%d, %d, %d)\n",
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("  Memory Bus Width          : %d bits\n", prop.memoryBusWidth);
            printf("  L2 Cache Size             : %d KB\n", prop.l2CacheSize / 1024);
            printf("  Shared Memory per Block   : %lu KB\n", prop.sharedMemPerBlock / 1024);
            printf("  Total Constant Memory     : %lu KB\n", prop.totalConstMem / 1024);
            printf("  Device Can Map Host Memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
            printf("  Concurrent Kernels        : %s\n", prop.concurrentKernels ? "Yes" : "No");
            printf("  ECC Enabled               : %s\n", prop.ECCEnabled ? "Yes" : "No");
            printf("  Unified Addressing        : %s\n", prop.unifiedAddressing ? "Yes" : "No");
            printf("====================================================================\n");
        }
    }

    return 0;
}
