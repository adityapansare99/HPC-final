#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 8000

void init_data(double *A, double *x, double *y_seq, double *y_par) {
    for (int i = 0; i < N; i++) {
        x[i] = (double)(i % 100) / 10.0;
        y_seq[i] = 0.0;
        y_par[i] = 0.0;
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (double)((i + j) % 100) / 10.0;
        }
    }
}

int verify_results(double *y_seq, double *y_par) {
    for (int i = 0; i < N; i++) {
        double diff = y_seq[i] - y_par[i];
        if (diff < 0) diff = -diff;
        if (diff > 1e-5) {
            printf("Mismatch at index %d: Seq=%f, Par=%f\n", i, y_seq[i], y_par[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int requested_threads = 4;
    if (argc > 1) {
        requested_threads = atoi(argv[1]);
    }

    printf("========================================================================\n");
    printf("   Assignment 4 Task 1: Matrix-Vector Multiplication (Input Decomposition)\n");
    printf("========================================================================\n");
    printf("Matrix Dimension: %d x %d | Vector Length: %d\n", N, N, N);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));
    double *y_seq = (double *)malloc(N * sizeof(double));
    double *y_par = (double *)malloc(N * sizeof(double));

    if (!A || !x || !y_seq || !y_par) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    init_data(A, x, y_seq, y_par);

    // Sequential Matrix-Vector Multiplication
    double start_seq = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y_seq[i] = sum;
    }
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;
    printf("Sequential Execution Time: %.6f seconds\n\n", seq_time);

    // OpenMP Parallel Matrix-Vector Multiplication (Input Data Decomposition - Row Partitioning)
    int test_threads[] = {1, 2, 4, 8, requested_threads};
    int num_tests = sizeof(test_threads) / sizeof(test_threads[0]);

    printf("%-10s %-18s %-12s %-12s %-12s\n", "Threads", "Execution Time (s)", "Speedup", "Efficiency", "Verified");
    printf("------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        int threads = test_threads[t];
        omp_set_num_threads(threads);

        double start_par = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A[i * N + j] * x[j];
            }
            y_par[i] = sum;
        }
        double end_par = omp_get_wtime();
        double par_time = end_par - start_par;

        double speedup = seq_time / par_time;
        double efficiency = speedup / threads;
        int verified = verify_results(y_seq, y_par);

        printf("%-10d %-18.6f %-12.4f %-12.4f %-12s\n",
               threads, par_time, speedup, efficiency, verified ? "PASSED" : "FAILED");
    }

    free(A);
    free(x);
    free(y_seq);
    free(y_par);

    return 0;
}
