/*
 * ==============================================================================
 * HPC Experiment 10 - Task 3: Global Memory vs Shared Memory MatMul Comparison
 * ==============================================================================

 * Description:
 *   Systematic comparison of Global Memory vs Shared Memory Tiled Matrix
 *   Multiplication across matrix dimensions:
 *     256x256, 512x512, 1024x1024
 *   and block/tile dimensions:
 *     8x8, 16x16, 32x32.
 *   Calculates Speedup of Shared Memory relative to Global Memory.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

__global__ void matmulGlobal(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <int TILE>
__global__ void matmulShared(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[TILE][TILE];
    __shared__ float s_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int m = 0; m < numTiles; ++m) {
        if (row < N && (m * TILE + threadIdx.x) < N)
            s_A[threadIdx.y][threadIdx.x] = A[row * N + m * TILE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;

        if ((m * TILE + threadIdx.y) < N && col < N)
            s_B[threadIdx.y][threadIdx.x] = B[(m * TILE + threadIdx.y) * N + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    printf("========================================================================================\n");
    printf(" HPC Experiment 10 - Task 3: Global Memory vs Shared Memory Performance Comparison\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("========================================================================================\n\n");

    int sizes[] = {256, 512, 1024};
    int tiles[] = {8, 16, 32};

    printf("+------------+------------+--------------------+--------------------+-------------------+\n");
    printf("| MatrixSize | Block/Tile | Global Memory Time | Shared Memory Time | Speedup (Shared)  |\n");
    printf("+------------+------------+--------------------+--------------------+-------------------+\n");

    for (int s = 0; s < 3; ++s) {
        int N = sizes[s];
        size_t bytes = (size_t)N * N * sizeof(float);
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = 1.0f;
            h_B[i] = 2.0f;
        }

        float *d_A, *d_B, *d_C_glob, *d_C_sh;
        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_C_glob, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_C_sh, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        for (int t = 0; t < 3; ++t) {
            int T = tiles[t];
            dim3 block(T, T);
            dim3 grid((N + T - 1) / T, (N + T - 1) / T);

            cudaEvent_t startG, stopG, startS, stopS;
            CHECK_CUDA(cudaEventCreate(&startG));
            CHECK_CUDA(cudaEventCreate(&stopG));
            CHECK_CUDA(cudaEventCreate(&startS));
            CHECK_CUDA(cudaEventCreate(&stopS));

            // Warm-up
            matmulGlobal<<<grid, block>>>(d_A, d_B, d_C_glob, N);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Global Timing
            CHECK_CUDA(cudaEventRecord(startG));
            for (int r = 0; r < 5; ++r) {
                matmulGlobal<<<grid, block>>>(d_A, d_B, d_C_glob, N);
            }
            CHECK_CUDA(cudaEventRecord(stopG));
            CHECK_CUDA(cudaEventSynchronize(stopG));
            float timeG = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&timeG, startG, stopG));
            timeG /= 5.0f;

            // Shared Timing
            CHECK_CUDA(cudaEventRecord(startS));
            for (int r = 0; r < 5; ++r) {
                if (T == 8) matmulShared<8><<<grid, block>>>(d_A, d_B, d_C_sh, N);
                else if (T == 16) matmulShared<16><<<grid, block>>>(d_A, d_B, d_C_sh, N);
                else if (T == 32) matmulShared<32><<<grid, block>>>(d_A, d_B, d_C_sh, N);
            }
            CHECK_CUDA(cudaEventRecord(stopS));
            CHECK_CUDA(cudaEventSynchronize(stopS));
            float timeS = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&timeS, startS, stopS));
            timeS /= 5.0f;

            float speedup = timeG / timeS;
            printf("| %4d x %4d |   %2d x %2d  |     %8.3f ms    |     %8.3f ms    |       %6.2fx      |\n",
                   N, N, T, T, timeG, timeS, speedup);

            printf("RESULT_JSON:{\"type\":\"task3_compare\",\"matrix_size\":%d,\"tile_size\":%d,\"global_time_ms\":%.4f,\"shared_time_ms\":%.4f,\"speedup\":%.2f}\n",
                   N, T, timeG, timeS, speedup);

            CHECK_CUDA(cudaEventDestroy(startG));
            CHECK_CUDA(cudaEventDestroy(stopG));
            CHECK_CUDA(cudaEventDestroy(startS));
            CHECK_CUDA(cudaEventDestroy(stopS));
        }
        printf("+------------+------------+--------------------+--------------------+-------------------+\n");

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C_glob));
        CHECK_CUDA(cudaFree(d_C_sh));
        free(h_A);
        free(h_B);
    }

    return 0;
}
