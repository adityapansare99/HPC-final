/*
 * ==============================================================================
 * HPC Experiment 10 - Challenge 4: 2D Image Convolution using Shared Memory
 * ==============================================================================

 * Description:
 *   Implements 2D Convolution (e.g. 5x5 Gaussian blur / edge filter) across:
 *     1. Sequential CPU implementation.
 *     2. CUDA Global Memory kernel (each thread reads 25 neighboring elements from DRAM).
 *     3. CUDA Shared Memory kernel with Halo Border Caching:
 *        Threads collaboratively load an (TILE + 2*RADIUS) x (TILE + 2*RADIUS)
 *        patch into fast shared memory, synchronizing with __syncthreads().
 *   Benchmarked on 1024x1024 and 2048x2048 image matrices.
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

#define RADIUS 2
#define KERNEL_SIZE (2 * RADIUS + 1) // 5x5 filter
#define TILE_DIM 16
#define SHARED_DIM (TILE_DIM + 2 * RADIUS) // 20x20

/* Constant Memory for 5x5 Convolution Mask */
__constant__ float d_Filter[KERNEL_SIZE * KERNEL_SIZE];

/* CPU 2D Convolution */
void conv2dCPU(const float *in, const float *mask, float *out, int W, int H) {
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float sum = 0.0f;
            for (int kr = -RADIUS; kr <= RADIUS; ++kr) {
                for (int kc = -RADIUS; kc <= RADIUS; ++kc) {
                    int in_r = r + kr;
                    int in_c = c + kc;
                    // Clamp to boundary
                    if (in_r < 0) in_r = 0;
                    if (in_r >= H) in_r = H - 1;
                    if (in_c < 0) in_c = 0;
                    if (in_c >= W) in_c = W - 1;

                    float m_val = mask[(kr + RADIUS) * KERNEL_SIZE + (kc + RADIUS)];
                    sum += in[in_r * W + in_c] * m_val;
                }
            }
            out[r * W + c] = sum;
        }
    }
}

/* CUDA Global Memory Convolution */
__global__ void conv2dGlobalKernel(const float *in, float *out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < H && col < W) {
        float sum = 0.0f;
        for (int kr = -RADIUS; kr <= RADIUS; ++kr) {
            for (int kc = -RADIUS; kc <= RADIUS; ++kc) {
                int in_r = min(max(row + kr, 0), H - 1);
                int in_c = min(max(col + kc, 0), W - 1);
                float m_val = d_Filter[(kr + RADIUS) * KERNEL_SIZE + (kc + RADIUS)];
                sum += in[in_r * W + in_c] * m_val;
            }
        }
        out[row * W + col] = sum;
    }
}

/* CUDA Shared Memory Convolution with Halo Cells */
__global__ void conv2dSharedKernel(const float *in, float *out, int W, int H) {
    __shared__ float s_data[SHARED_DIM][SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_col = blockIdx.x * TILE_DIM + tx;
    int out_row = blockIdx.y * TILE_DIM + ty;

    // Collaborative loading of shared memory tile including halo cells
    // Stride loops to populate SHARED_DIM x SHARED_DIM using TILE_DIM x TILE_DIM threads
    for (int sy = ty; sy < SHARED_DIM; sy += TILE_DIM) {
        for (int sx = tx; sx < SHARED_DIM; sx += TILE_DIM) {
            int in_r = blockIdx.y * TILE_DIM + sy - RADIUS;
            int in_c = blockIdx.x * TILE_DIM + sx - RADIUS;

            in_r = min(max(in_r, 0), H - 1);
            in_c = min(max(in_c, 0), W - 1);

            s_data[sy][sx] = in[in_r * W + in_c];
        }
    }
    __syncthreads();

    if (out_row < H && out_col < W) {
        float sum = 0.0f;
        #pragma unroll
        for (int kr = -RADIUS; kr <= RADIUS; ++kr) {
            #pragma unroll
            for (int kc = -RADIUS; kc <= RADIUS; ++kc) {
                float m_val = d_Filter[(kr + RADIUS) * KERNEL_SIZE + (kc + RADIUS)];
                sum += s_data[ty + RADIUS + kr][tx + RADIUS + kc] * m_val;
            }
        }
        out[out_row * W + out_col] = sum;
    }
}

int main() {
    printf("========================================================================================\n");
    printf(" HPC Experiment 10 - Challenge 4: 2D Convolution (CPU vs CUDA Global vs CUDA Shared)\n");
    printf(" High Performance Computing | PRN: 23510080\n");
    printf("========================================================================================\n\n");

    // 5x5 Gaussian blur filter
    float h_Filter[25] = {
        1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256,
        4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
        6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
        4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
        1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256
    };
    CHECK_CUDA(cudaMemcpyToSymbol(d_Filter, h_Filter, 25 * sizeof(float)));

    int dims[] = {1024, 2048};

    for (int d = 0; d < 2; ++d) {
        int W = dims[d], H = dims[d];
        size_t bytes = (size_t)W * H * sizeof(float);
        float *h_in = (float*)malloc(bytes);
        float *h_cpu = (float*)malloc(bytes);
        float *h_gpu = (float*)malloc(bytes);

        for (int i = 0; i < W * H; ++i) h_in[i] = sinf((float)i * 0.01f);

        auto t0 = std::chrono::high_resolution_clock::now();
        conv2dCPU(h_in, h_Filter, h_cpu, W, H);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_out, bytes));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);

        cudaEvent_t startG, stopG, startS, stopS;
        CHECK_CUDA(cudaEventCreate(&startG));
        CHECK_CUDA(cudaEventCreate(&stopG));
        CHECK_CUDA(cudaEventCreate(&startS));
        CHECK_CUDA(cudaEventCreate(&stopS));

        // Global Kernel
        CHECK_CUDA(cudaEventRecord(startG));
        for (int r = 0; r < 5; ++r) {
            conv2dGlobalKernel<<<grid, block>>>(d_in, d_out, W, H);
        }
        CHECK_CUDA(cudaEventRecord(stopG));
        CHECK_CUDA(cudaEventSynchronize(stopG));
        float glob_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&glob_ms, startG, stopG));
        glob_ms /= 5.0f;

        // Shared Kernel
        CHECK_CUDA(cudaEventRecord(startS));
        for (int r = 0; r < 5; ++r) {
            conv2dSharedKernel<<<grid, block>>>(d_in, d_out, W, H);
        }
        CHECK_CUDA(cudaEventRecord(stopS));
        CHECK_CUDA(cudaEventSynchronize(stopS));
        float sh_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&sh_ms, startS, stopS));
        sh_ms /= 5.0f;

        CHECK_CUDA(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

        // Verify
        float max_diff = 0.0f;
        for (int i = 0; i < W * H; ++i) {
            float diff = fabsf(h_cpu[i] - h_gpu[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("  Image Size: %4d x %4d | Max Diff: %.2e | CPU: %8.2f ms | CUDA Global: %6.2f ms | CUDA Shared: %6.2f ms | Speedup (Shared vs CPU): %6.2fx | Speedup (Shared vs Glob): %5.2fx\n",
               W, H, max_diff, cpu_ms, glob_ms, sh_ms, cpu_ms / sh_ms, glob_ms / sh_ms);

        printf("RESULT_JSON:{\"type\":\"challenge4_conv2d\",\"size\":%d,\"cpu_ms\":%.4f,\"global_ms\":%.4f,\"shared_ms\":%.4f,\"speedup_vs_cpu\":%.2f,\"speedup_vs_global\":%.2f}\n",
               W, cpu_ms, glob_ms, sh_ms, cpu_ms / sh_ms, glob_ms / sh_ms);

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        CHECK_CUDA(cudaEventDestroy(startG));
        CHECK_CUDA(cudaEventDestroy(stopG));
        CHECK_CUDA(cudaEventDestroy(startS));
        CHECK_CUDA(cudaEventDestroy(stopS));
        free(h_in);
        free(h_cpu);
        free(h_gpu);
    }

    return 0;
}
