#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void run_benchmarks() {
    long N = 50000000L;
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    printf("=========================================================\n");
    printf("  Assignment 3: Accurate Benchmarks (N = %ld)\n", N);
    printf("=========================================================\n");

    int threads[] = {1, 2, 4, 8};

    // Array Sum / Dot Product / Prefix Sum timing
    for (int t = 0; t < 4; t++) {
        int P = threads[t];
        omp_set_num_threads(P);

        // Dot Product
        double t0 = omp_get_wtime();
        double dot = 0.0;
        #pragma omp parallel for reduction(+:dot)
        for (long i = 0; i < N; i++) {
            dot += A[i] * B[i];
        }
        double dot_time = omp_get_wtime() - t0;

        // Prefix Sum (Parallel 2-pass)
        double *out = (double *)malloc(N * sizeof(double));
        double *offset = (double *)malloc(P * sizeof(double));
        t0 = omp_get_wtime();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long chunk = N / P;
            long start = tid * chunk;
            long end = (tid == P - 1) ? N : (tid + 1) * chunk;
            out[start] = A[start];
            for (long i = start + 1; i < end; i++) out[i] = out[i - 1] + A[i];
            offset[tid] = out[end - 1];
            #pragma omp barrier
            #pragma omp single
            {
                for (int i = 1; i < P; i++) offset[i] += offset[i - 1];
            }
            if (tid > 0) {
                double add = offset[tid - 1];
                for (long i = start; i < end; i++) out[i] += add;
            }
        }
        double pref_time = omp_get_wtime() - t0;
        free(out);
        free(offset);

        printf("Threads = %d | Dot Product Time = %.6fs | Prefix Sum Time = %.6fs\n", P, dot_time, pref_time);
    }

    free(A);
    free(B);
}

int main() {
    run_benchmarks();
    return 0;
}
