#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_mult_static(int N, double *A, double *B, double *C, int num_threads) {
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

void matrix_mult_dynamic(int N, double *A, double *B, double *C, int num_threads) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void matrix_mult_guided(int N, double *A, double *B, double *C, int num_threads) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(guided, 16)
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
    printf("  Task 4: OpenMP Scheduling Strategies Analysis           \n");
    printf("=========================================================\n");

    int N = 1000;
    int num_threads = 4;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    printf("Matrix Size: %d x %d | Threads: %d\n\n", N, N, num_threads);

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
    printf("Sequential Baseline Execution Time: %.6f seconds\n\n", seq_time);

    printf("%-20s | %-18s | %-12s\n", "Scheduling Type", "Execution Time (s)", "Speedup");
    printf("-------------------------------------------------------\n");

    t_start = omp_get_wtime();
    matrix_mult_static(N, A, B, C, num_threads);
    double static_time = omp_get_wtime() - t_start;
    if (static_time < 0.0001) static_time = 0.0001;
    printf("%-20s | %-18.6f | %-12.4fx\n", "Static", static_time, seq_time / static_time);

    t_start = omp_get_wtime();
    matrix_mult_dynamic(N, A, B, C, num_threads);
    double dynamic_time = omp_get_wtime() - t_start;
    if (dynamic_time < 0.0001) dynamic_time = 0.0001;
    printf("%-20s | %-18.6f | %-12.4fx\n", "Dynamic (chunk=16)", dynamic_time, seq_time / dynamic_time);

    t_start = omp_get_wtime();
    matrix_mult_guided(N, A, B, C, num_threads);
    double guided_time = omp_get_wtime() - t_start;
    if (guided_time < 0.0001) guided_time = 0.0001;
    printf("%-20s | %-18.6f | %-12.4fx\n", "Guided (chunk=16)", guided_time, seq_time / guided_time);

    free(A);
    free(B);
    free(C);
    return 0;
}

// gcc -fopenmp -O2 task4_scheduling.c -o task4.exe ; 
// ./task4.exe
