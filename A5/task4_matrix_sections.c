#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mat_add(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = A[i] + B[i];
}

void mat_sub(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = A[i] - B[i];
}

void mat_transpose(double *A, double *T, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T[j * N + i] = A[i * N + j];
        }
    }
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Task 4: Independent Matrix Operations (OpenMP Sections)\n");
    printf("========================================================================\n\n");

    int sizes[] = {500, 1000, 1500};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-15s %-20s %-20s %-12s\n", "Matrix Size", "Sequential Time (s)", "Sections Time (s)", "Speedup");
    printf("------------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        size_t bytes = (size_t)N * N * sizeof(double);

        double *A = (double *)malloc(bytes);
        double *B = (double *)malloc(bytes);
        double *C_add = (double *)malloc(bytes);
        double *C_sub = (double *)malloc(bytes);
        double *T_A = (double *)malloc(bytes);
        double *T_B = (double *)malloc(bytes);

        for (int i = 0; i < N * N; i++) {
            A[i] = (double)(i % 100) / 10.0;
            B[i] = (double)(i % 50) / 10.0;
        }

        // Sequential Execution
        double t0 = omp_get_wtime();
        mat_add(A, B, C_add, N);
        mat_sub(A, B, C_sub, N);
        mat_transpose(A, T_A, N);
        mat_transpose(B, T_B, N);
        double t_seq = omp_get_wtime() - t0;

        // OpenMP Sections Execution
        omp_set_num_threads(4);
        t0 = omp_get_wtime();
        #pragma omp parallel sections
        {
            #pragma omp section
            mat_add(A, B, C_add, N);

            #pragma omp section
            mat_sub(A, B, C_sub, N);

            #pragma omp section
            mat_transpose(A, T_A, N);

            #pragma omp section
            mat_transpose(B, T_B, N);
        }
        double t_sec = omp_get_wtime() - t0;

        printf("%-15dx%-14d %-20.6f %-20.6f %-12.2fx\n", N, N, t_seq, t_sec, t_seq / t_sec);

        free(A); free(B); free(C_add); free(C_sub); free(T_A); free(T_B);
    }

    return 0;
}
