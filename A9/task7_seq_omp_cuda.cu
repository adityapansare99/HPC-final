/*
 * ==============================================================================
 * HPC Experiment 9 - Task 7: Sequential vs OpenMP vs CUDA Vector Addition
 * ==============================================================================

 * Description:
 *   Implements vector addition using three approaches:
 *     Version 1: Sequential C/C++
 *     Version 2: OpenMP (shared-memory parallelism)
 *     Version 3: CUDA (GPU parallelism)
 *   Compares execution time across vector sizes: 10^5, 10^6, 10^7, 10^8.
 *   Verifies that all three produce identical results.
 *
 * Compile: nvcc -Xcompiler -fopenmp task7_seq_omp_cuda.cu -o task7_seq_omp_cuda -lm
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

/* ---- Sequential Version ---- */
void vectorAddSeq(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

/* ---- OpenMP Version ---- */
void vectorAddOMP(const float *A, const float *B, float *C, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

/* ---- CUDA Kernel ---- */
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}

/* ---- Verify results match ---- */
int verifyAll(const float *C_seq, const float *C_omp, const float *C_cuda, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(C_seq[i] - C_omp[i]) > 1e-4f ||
            fabsf(C_seq[i] - C_cuda[i]) > 1e-4f) {
            errors++;
        }
    }
    return errors;
}

int main(int argc, char *argv[]) {
    int benchmark_mode = 0;
    if (argc > 1) benchmark_mode = atoi(argv[1]);

    long sizes[] = {100000L, 1000000L, 10000000L, 100000000L};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int threads_per_block = 256;

    if (!benchmark_mode) {
        printf("====================================================================\n");
        printf("  Task 7: Vector Addition — Sequential vs OpenMP vs CUDA\n");
        printf("  OpenMP Threads: %d\n", omp_get_max_threads());
        printf("====================================================================\n");
        printf("  %-12s %-12s %-12s %-12s %-10s %-10s %s\n",
               "Size", "Seq (ms)", "OMP (ms)", "CUDA (ms)",
               "OMP Spdup", "CUDA Spdup", "Verify");
        printf("  %-12s %-12s %-12s %-12s %-10s %-10s %s\n",
               "----", "--------", "--------", "---------",
               "---------", "----------", "------");
    }

    for (int s = 0; s < num_sizes; s++) {
        long N = sizes[s];
        size_t size = N * sizeof(float);

        /* Check available memory - skip sizes that are too large */
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C_seq  = (float *)malloc(size);
        float *h_C_omp  = (float *)malloc(size);
        float *h_C_cuda = (float *)malloc(size);

        if (!h_A || !h_B || !h_C_seq || !h_C_omp || !h_C_cuda) {
            if (!benchmark_mode)
                printf("  %-12ld Skipped (insufficient host memory)\n", N);
            free(h_A); free(h_B); free(h_C_seq); free(h_C_omp); free(h_C_cuda);
            continue;
        }

        /* Initialize */
        for (long i = 0; i < N; i++) {
            h_A[i] = sinf(i * 0.0001f) * 50.0f;
            h_B[i] = cosf(i * 0.0001f) * 50.0f;
        }

        /* ---- Version 1: Sequential ---- */
        auto ts1 = std::chrono::high_resolution_clock::now();
        vectorAddSeq(h_A, h_B, h_C_seq, N);
        auto te1 = std::chrono::high_resolution_clock::now();
        double seq_ms = std::chrono::duration<double, std::milli>(te1 - ts1).count();

        /* ---- Version 2: OpenMP ---- */
        auto ts2 = std::chrono::high_resolution_clock::now();
        vectorAddOMP(h_A, h_B, h_C_omp, N);
        auto te2 = std::chrono::high_resolution_clock::now();
        double omp_ms = std::chrono::duration<double, std::milli>(te2 - ts2).count();

        /* ---- Version 3: CUDA ---- */
        float *d_A, *d_B, *d_C;
        cudaError_t err1 = cudaMalloc((void **)&d_A, size);
        cudaError_t err2 = cudaMalloc((void **)&d_B, size);
        cudaError_t err3 = cudaMalloc((void **)&d_C, size);

        double cuda_ms = -1.0;
        int verify_errors = 0;

        if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess) {
            int num_blocks = (N + threads_per_block - 1) / threads_per_block;

            cudaEvent_t ev_start, ev_end;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_end);

            cudaEventRecord(ev_start);
            cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
            vectorAddKernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
            cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);
            cudaEventRecord(ev_end);
            cudaEventSynchronize(ev_end);

            float elapsed = 0;
            cudaEventElapsedTime(&elapsed, ev_start, ev_end);
            cuda_ms = (double)elapsed;

            verify_errors = verifyAll(h_C_seq, h_C_omp, h_C_cuda, N);

            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_end);
        } else {
            if (!benchmark_mode)
                printf("  %-12ld GPU memory insufficient, CUDA skipped.\n", N);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            free(h_A); free(h_B); free(h_C_seq); free(h_C_omp); free(h_C_cuda);
            continue;
        }

        double omp_speedup  = seq_ms / omp_ms;
        double cuda_speedup = seq_ms / cuda_ms;

        if (benchmark_mode) {
            printf("DATA_COMPARE|%ld|%.4f|%.4f|%.4f|%.4f|%.4f|%d\n",
                   N, seq_ms, omp_ms, cuda_ms, omp_speedup, cuda_speedup, verify_errors);
        } else {
            printf("  %-12ld %-12.4f %-12.4f %-12.4f %-10.2f %-10.2f %s\n",
                   N, seq_ms, omp_ms, cuda_ms, omp_speedup, cuda_speedup,
                   verify_errors == 0 ? "✓ PASS" : "✗ FAIL");
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_seq);
        free(h_C_omp);
        free(h_C_cuda);
    }

    if (!benchmark_mode) {
        printf("====================================================================\n");
    }

    return 0;
}
