#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_mult_standard(int N, double *A, double *B, double *C, int num_threads) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void matrix_mult_collapse2(int N, double *A, double *B, double *C, int num_threads) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Task 5: Optimization of Parallel Matrix Multiplication \n");
    printf("=========================================================\n");

    int N = 1000;
    if (argc > 1) N = atoi(argv[1]);

    int max_threads = omp_get_max_threads();
    printf("Matrix Size: %d x %d | System Max Threads: %d\n", N, N, max_threads);

    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    double t_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
    double seq_time = omp_get_wtime() - t_start;
    if (seq_time < 0.001) seq_time = 0.001;
    printf("Sequential Execution Time (Ts): %.6f seconds\n\n", seq_time);

    int thread_counts[] = {1, 2, 4, 8, max_threads};
    int num_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("--- Performance Comparison across Thread Counts (collapse(2)) ---\n");
    printf("%-10s | %-18s | %-12s | %-12s\n", "Threads", "Execution Time (s)", "Speedup", "Efficiency");
    printf("-----------------------------------------------------------------\n");

    for (int i = 0; i < num_counts; i++) {
        int threads = thread_counts[i];
        if (i == num_counts - 1 && (threads == 1 || threads == 2 || threads == 4 || threads == 8)) {
            continue;
        }

        t_start = omp_get_wtime();
        matrix_mult_collapse2(N, A, B, C, threads);
        double par_time = omp_get_wtime() - t_start;
        if (par_time < 0.0001) par_time = 0.0001;

        double speedup = seq_time / par_time;
        double efficiency = speedup / threads;

        printf("%-10d | %-18.6f | %-12.4fx | %-12.4f\n", threads, par_time, speedup, efficiency);
    }

    printf("\n--- Loop Collapsing Impact Analysis (Threads = 4) ---\n");
    t_start = omp_get_wtime();
    matrix_mult_standard(N, A, B, C, 4);
    double std_time = omp_get_wtime() - t_start;
    if (std_time < 0.0001) std_time = 0.0001;

    t_start = omp_get_wtime();
    matrix_mult_collapse2(N, A, B, C, 4);
    double col2_time = omp_get_wtime() - t_start;
    if (col2_time < 0.0001) col2_time = 0.0001;

    printf("Standard Parallel loop: %.6f seconds\n", std_time);
    printf("Collapse(2) Parallel  : %.6f seconds (Speedup vs Standard: %.2fx)\n",
           col2_time, std_time / col2_time);

    free(A);
    free(B);
    free(C);

    return 0;
}

// gcc -fopenmp -O2 task5_optimization.c -o task5.exe ; 
// ./task5.exe