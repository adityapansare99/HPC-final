#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void init_matrices(int N, double *A, double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((i + 2 * j) % 15 + 1) * 0.1;
            B[i * N + j] = ((2 * i + j) % 15 + 1) * 0.05;
        }
    }
}

void sequential_matmul(int N, const double *A, const double *B, double *C) {
    for (int i = 0; i < N; i++) {
        int i_off = i * N;
        for (int j = 0; j < N; j++) {
            C[i_off + j] = 0.0;
        }
        for (int k = 0; k < N; k++) {
            double a_ik = A[i_off + k];
            int k_off = k * N;
            for (int j = 0; j < N; j++) {
                C[i_off + j] += a_ik * B[k_off + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 500;
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    size_t total_elements = (size_t)N * N;
    double *A = (double *)malloc(total_elements * sizeof(double));
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C = (double *)malloc(total_elements * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for size %d x %d\n", N, N);
        return 1;
    }

    init_matrices(N, A, B);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sequential_matmul(N, A, B, C);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    double checksum = 0.0;
    for (size_t i = 0; i < total_elements; i += N + 1) checksum += C[i]; 

    if (benchmark_mode) {
        printf("DATA_SEQ_MATMUL|%d|1|%.4f|%.4f\n", N, elapsed_ms, checksum);
    } else {
        printf("=================================================================\n");
        printf("  Task 1A: Sequential Matrix-Matrix Multiplication (C = A * B)   \n");
        printf("=================================================================\n");
        printf("Matrix Dimension : %d x %d (%zu elements)\n", N, N, total_elements);
        printf("Execution Time   : %.4f ms (%.4f seconds)\n", elapsed_ms, elapsed_ms / 1000.0);
        printf("Diagonal Checksum: %.6f\n", checksum);
        printf("=================================================================\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
