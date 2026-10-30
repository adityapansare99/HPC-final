#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void run_prefix_sum_experiment(long N, int num_threads) {
    double *in = (double *)malloc(N * sizeof(double));
    double *out_seq = (double *)malloc(N * sizeof(double));
    double *out_par = (double *)malloc(N * sizeof(double));

    if (!in || !out_seq || !out_par) {
        printf("Memory allocation failed!\n");
        return;
    }

    for (long i = 0; i < N; i++) {
        in[i] = 1.0;
    }

    // 1. Sequential Prefix Sum (Inclusive Scan)
    double start_seq = omp_get_wtime();
    out_seq[0] = in[0];
    for (long i = 1; i < N; i++) {
        out_seq[i] = out_seq[i - 1] + in[i];
    }
    double time_seq = omp_get_wtime() - start_seq;

    // 2. Parallel Prefix Sum (Two-Pass Chunk-Based Scan)
    omp_set_num_threads(num_threads);
    double start_par = omp_get_wtime();
    double *offset = (double *)malloc(num_threads * sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int P = omp_get_num_threads();
        long chunk = N / P;
        long start_idx = tid * chunk;
        long end_idx = (tid == P - 1) ? N : (tid + 1) * chunk;

        // Phase 1: Local prefix sum per chunk
        out_par[start_idx] = in[start_idx];
        for (long i = start_idx + 1; i < end_idx; i++) {
            out_par[i] = out_par[i - 1] + in[i];
        }
        offset[tid] = out_par[end_idx - 1];

        #pragma omp barrier

        // Phase 2: Compute prefix sum of chunk totals (Sequential scan over P chunks)
        #pragma omp single
        {
            for (int i = 1; i < P; i++) {
                offset[i] += offset[i - 1];
            }
        }

        // Phase 3: Add chunk offsets to remaining blocks
        if (tid > 0) {
            double add_val = offset[tid - 1];
            for (long i = start_idx; i < end_idx; i++) {
                out_par[i] += add_val;
            }
        }
    }
    double time_par = omp_get_wtime() - start_par;
    free(offset);

    // Verify correctness
    int correct = 1;
    for (long i = 0; i < N; i++) {
        if (out_seq[i] != out_par[i]) {
            correct = 0;
            break;
        }
    }

    double speedup = time_seq / time_par;
    double efficiency = speedup / num_threads;

    printf("N = %10ld | Threads = %2d | Seq: %.6fs | Par: %.6fs | Speedup: %.2fx | Efficiency: %.2f | Status: %s\n",
           N, num_threads, time_seq, time_par, speedup, efficiency, correct ? "PASSED" : "FAILED");

    free(in);
    free(out_seq);
    free(out_par);
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Assignment 3 - Task 3: Parallel Prefix Sum (Scan)      \n");
    printf("=========================================================\n");

    long N = 10000000L;
    int thread_counts[] = {1, 2, 4, 8};
    int num_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_counts; i++) {
        run_prefix_sum_experiment(N, thread_counts[i]);
    }

    return 0;
}
