#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void init_matrices(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = (double)((i + j) % 100) / 10.0;
            B[i * size + j] = (double)((i - j + 100) % 100) / 10.0;
            C[i * size + j] = 0.0;
        }
    }
}

void matmul_seq(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void matmul_omp_static(double *A, double *B, double *C, int size, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void matmul_omp_dynamic(double *A, double *B, double *C, int size, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int verify_matrix(double *C1, double *C2, int size) {
    for (int i = 0; i < size * size; i++) {
        double diff = fabs(C1[i] - C2[i]);
        if (diff > 1e-4) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }

    printf("========================================================================\n");
    printf("   Assignment 4 Task 2: Output Data Decomposition (Matrix-Matrix Mult)  \n");
    printf("========================================================================\n");

    int sizes[] = {250, 500, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("\n>>> Testing Matrix Size: %d x %d (Threads = %d) <<<\n", N, N, num_threads);

        double *A = (double *)malloc((size_t)N * N * sizeof(double));
        double *B = (double *)malloc((size_t)N * N * sizeof(double));
        double *C_seq = (double *)malloc((size_t)N * N * sizeof(double));
        double *C_stat = (double *)malloc((size_t)N * N * sizeof(double));
        double *C_dyn = (double *)malloc((size_t)N * N * sizeof(double));

        init_matrices(A, B, C_seq, N);

        // Sequential
        double t0 = omp_get_wtime();
        matmul_seq(A, B, C_seq, N);
        double t_seq = omp_get_wtime() - t0;

        // Static scheduling
        init_matrices(A, B, C_stat, N);
        t0 = omp_get_wtime();
        matmul_omp_static(A, B, C_stat, N, num_threads);
        double t_stat = omp_get_wtime() - t0;

        // Dynamic scheduling
        init_matrices(A, B, C_dyn, N);
        t0 = omp_get_wtime();
        matmul_omp_dynamic(A, B, C_dyn, N, num_threads);
        double t_dyn = omp_get_wtime() - t0;

        int ok_stat = verify_matrix(C_seq, C_stat, N);
        int ok_dyn = verify_matrix(C_seq, C_dyn, N);

        printf("Sequential Execution Time : %.6f s\n", t_seq);
        printf("OpenMP Static Time        : %.6f s | Speedup: %.2fx | Verified: %s\n",
               t_stat, t_seq / t_stat, ok_stat ? "PASSED" : "FAILED");
        printf("OpenMP Dynamic Time       : %.6f s | Speedup: %.2fx | Verified: %s\n",
               t_dyn, t_seq / t_dyn, ok_dyn ? "PASSED" : "FAILED");

        free(A); free(B); free(C_seq); free(C_stat); free(C_dyn);
    }

    return 0;
}
