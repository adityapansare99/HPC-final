#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 800

void init_matrices(double *A, double *B, double *C) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)(i % 100) / 10.0;
        B[i] = (double)(i % 70) / 10.0;
        C[i] = 0.0;
    }
}

// Optimized i-k-j matrix multiplication for optimal L1/L2 cache locality
double run_matrix_mult_benchmark(int threads) {
    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    init_matrices(A, B, C);
    omp_set_num_threads(threads);

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double end = omp_get_wtime();

    free(A);
    free(B);
    free(C);

    return (end - start);
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 4 Task 4: Performance Analysis & Thread Scalability       \n");
    printf("========================================================================\n");
    printf("Matrix Dimension: %d x %d | Cache-Optimized i-k-j Kernel\n\n", N, N);

    int thread_counts[] = {1, 2, 4, 8, omp_get_max_threads()};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    double t1 = run_matrix_mult_benchmark(1);
    printf("1-Thread Baseline Time: %.6f seconds\n\n", t1);

    printf("%-12s %-20s %-15s %-15s\n", "Threads", "Execution Time (s)", "Speedup", "Efficiency");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        int p = thread_counts[i];
        double tp = (p == 1) ? t1 : run_matrix_mult_benchmark(p);
        double speedup = t1 / tp;
        double efficiency = speedup / p;

        printf("%-12d %-20.6f %-15.4f %-15.4f\n", p, tp, speedup, efficiency);
    }

    return 0;
}
