/*
 * ==============================================================================
 * HPC Experiment 10 - Task 4: Parallel Reduction (Sum, Max, Min)
 * ==============================================================================

 * Description:
 *   Implements parallel reduction to combine many values into a single scalar:
 *     Version A: Sequential CPU reduction (Sum, Max, Min).
 *     Version B: CUDA Global Memory reduction (direct global atomic updates).
 *     Version C: CUDA Shared Memory reduction (tree-based reduction in shared
 *                memory with __syncthreads(), then single atomic per block).
 *   Calculates:
 *     1. Sum of elements
 *     2. Maximum element
 *     3. Minimum element
 *   Tested across problem sizes: N = 10^5, 10^6, 10^7, 10^8.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
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

#define BLOCK_SIZE 256

/* Atomic Max for float */
__device__ static float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* Atomic Min for float */
__device__ static float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* -------------------------------------------------------------------------
 * CPU Implementations
 * ------------------------------------------------------------------------- */
void cpuReduction(const float *data, int N, double *sum_out, float *max_out, float *min_out) {
    double sum = 0.0;
    float max_v = -FLT_MAX;
    float min_v = FLT_MAX;
    for (int i = 0; i < N; ++i) {
        float val = data[i];
        sum += val;
        if (val > max_v) max_v = val;
        if (val < min_v) min_v = val;
    }
    *sum_out = sum;
    *max_out = max_v;
    *min_out = min_v;
}

/* -------------------------------------------------------------------------
 * Version B: CUDA Global Memory Reduction (Global Atomics for each element)
 * ------------------------------------------------------------------------- */
__global__ void reduceGlobalSumKernel(const float *data, int N, float *out_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(out_sum, data[idx]);
    }
}

__global__ void reduceGlobalMaxKernel(const float *data, int N, float *out_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicMaxFloat(out_max, data[idx]);
    }
}

__global__ void reduceGlobalMinKernel(const float *data, int N, float *out_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicMinFloat(out_min, data[idx]);
    }
}

/* -------------------------------------------------------------------------
 * Version C: CUDA Shared Memory Reduction (Tree Reduction per Block)
 * ------------------------------------------------------------------------- */
__global__ void reduceSharedSumKernel(const float *data, int N, float *out_sum) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load first element + second element (grid-stride / double load)
    float mySum = 0.0f;
    if (i < N) mySum += data[i];
    if (i + blockDim.x < N) mySum += data[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    // Tree reduction in shared memory (sequential addressing avoids bank conflicts)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 writes block sum to global memory
    if (tid == 0) {
        atomicAdd(out_sum, sdata[0]);
    }
}

__global__ void reduceSharedMaxKernel(const float *data, int N, float *out_max) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float myMax = -FLT_MAX;
    if (i < N) myMax = fmaxf(myMax, data[i]);
    if (i + blockDim.x < N) myMax = fmaxf(myMax, data[i + blockDim.x]);
    sdata[tid] = myMax;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(out_max, sdata[0]);
    }
}

__global__ void reduceSharedMinKernel(const float *data, int N, float *out_min) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float myMin = FLT_MAX;
    if (i < N) myMin = fminf(myMin, data[i]);
    if (i + blockDim.x < N) myMin = fminf(myMin, data[i + blockDim.x]);
    sdata[tid] = myMin;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinFloat(out_min, sdata[0]);
    }
}

void benchmarkReduction(int N) {
    printf("\n======================================================\n");
    printf(" Reduction Benchmark: N = %d elements (%.2f MB)\n", N, (N * sizeof(float)) / (1024.0 * 1024.0));
    printf("======================================================\n");

    size_t bytes = (size_t)N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)((i % 100) + 1); // values 1..100
    }

    /* 1. CPU Reference */
    double cpu_sum = 0.0;
    float cpu_max = 0.0f, cpu_min = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuReduction(h_data, N, &cpu_sum, &cpu_max, &cpu_min);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* 2. Allocate GPU buffers */
    float *d_data, *d_sum, *d_max, *d_min;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_sum, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_max, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_min, sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Transfer data to GPU
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    /* --- Version B: Global Memory Atomic Reduction --- */
    float init_zero = 0.0f, init_neg_inf = -FLT_MAX, init_pos_inf = FLT_MAX;
    CHECK_CUDA(cudaMemcpy(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, &init_neg_inf, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_min, &init_pos_inf, sizeof(float), cudaMemcpyHostToDevice));

    int grid_global = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaEventRecord(start));
    reduceGlobalSumKernel<<<grid_global, BLOCK_SIZE>>>(d_data, N, d_sum);
    reduceGlobalMaxKernel<<<grid_global, BLOCK_SIZE>>>(d_data, N, d_max);
    reduceGlobalMinKernel<<<grid_global, BLOCK_SIZE>>>(d_data, N, d_min);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float global_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&global_ms, start, stop));

    float gpu_glob_sum, gpu_glob_max, gpu_glob_min;
    CHECK_CUDA(cudaMemcpy(&gpu_glob_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_glob_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_glob_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));

    /* --- Version C: Shared Memory Tree Reduction --- */
    CHECK_CUDA(cudaMemcpy(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_max, &init_neg_inf, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_min, &init_pos_inf, sizeof(float), cudaMemcpyHostToDevice));

    int grid_shared = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    CHECK_CUDA(cudaEventRecord(start));
    reduceSharedSumKernel<<<grid_shared, BLOCK_SIZE>>>(d_data, N, d_sum);
    reduceSharedMaxKernel<<<grid_shared, BLOCK_SIZE>>>(d_data, N, d_max);
    reduceSharedMinKernel<<<grid_shared, BLOCK_SIZE>>>(d_data, N, d_min);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float shared_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, start, stop));

    float gpu_sh_sum, gpu_sh_max, gpu_sh_min;
    CHECK_CUDA(cudaMemcpy(&gpu_sh_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_sh_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&gpu_sh_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));

    printf("  CPU Execution Time      : %10.3f ms | Sum: %.1f, Max: %.1f, Min: %.1f\n",
           cpu_ms, cpu_sum, cpu_max, cpu_min);
    printf("  CUDA Global Memory Time : %10.3f ms | Sum: %.1f, Max: %.1f, Min: %.1f\n",
           global_ms, gpu_glob_sum, gpu_glob_max, gpu_glob_min);
    printf("  CUDA Shared Memory Time : %10.3f ms | Sum: %.1f, Max: %.1f, Min: %.1f\n",
           shared_ms, gpu_sh_sum, gpu_sh_max, gpu_sh_min);
    printf("  Speedup (Shared vs CPU) : %10.2fx\n", cpu_ms / shared_ms);
    printf("  Speedup (Shared vs Glob): %10.2fx\n", global_ms / shared_ms);

    printf("RESULT_JSON:{\"type\":\"reduction\",\"size\":%d,\"cpu_ms\":%.4f,\"global_ms\":%.4f,\"shared_ms\":%.4f,\"speedup_vs_cpu\":%.2f,\"speedup_vs_global\":%.2f}\n",
           N, cpu_ms, global_ms, shared_ms, cpu_ms / shared_ms, global_ms / shared_ms);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_sum));
    CHECK_CUDA(cudaFree(d_max));
    CHECK_CUDA(cudaFree(d_min));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_data);
}

int main(int argc, char **argv) {
    printf("====================================================================\n");
    printf(" CUDA Task 4: Parallel Reduction (Sum, Max, Min)\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("====================================================================\n");

    int test_sizes[] = {100000, 1000000, 10000000, 50000000};
    if (argc >= 2) {
        int custom_n = atoi(argv[1]);
        benchmarkReduction(custom_n);
        return 0;
    }

    for (int i = 0; i < 4; ++i) {
        benchmarkReduction(test_sizes[i]);
    }

    return 0;
}
