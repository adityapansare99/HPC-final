#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define DEFAULT_N 600

void init_data(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)(i % 50) + 1.0;
        B[i] = (double)(i % 30) + 1.0;
        C[i] = 0.0;
    }
}

// 1. Row-wise Decomposition (Outer loop over rows i is parallelized)
void matmul_rowwise(double *A, double *B, double *C, int N, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 2. Column-wise Decomposition (Middle loop over columns j is parallelized)
void matmul_colwise(double *A, double *B, double *C, int N, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// 3. Block Decomposition (2D grid of sub-blocks parallelized using collapse(2))
void matmul_block(double *A, double *B, double *C, int N, int threads) {
    omp_set_num_threads(threads);
    int block_size = 64; // Tile size for L1/L2 cache locality
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < N; bi += block_size) {
        for (int bj = 0; bj < N; bj += block_size) {
            for (int bk = 0; bk < N; bk += block_size) {
                int i_end = (bi + block_size < N) ? (bi + block_size) : N;
                int j_end = (bj + block_size < N) ? (bj + block_size) : N;
                int k_end = (bk + block_size < N) ? (bk + block_size) : N;
                
                for (int i = bi; i < i_end; i++) {
                    for (int j = bj; j < j_end; j++) {
                        double sum = C[i * N + j];
                        for (int k = bk; k < k_end; k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

int verify_results(double *C_ref, double *C_test, int N) {
    for (int i = 0; i < N * N; i++) {
        double diff = fabs(C_ref[i] - C_test[i]);
        if (diff > 1e-4) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int N = DEFAULT_N;
    int threads = 4;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    printf("========================================================================\n");
    printf("   Assignment 4 Task 3: Comparison of Decomposition Strategies          \n");
    printf("========================================================================\n");
    printf("Matrix Dimension: %d x %d | OpenMP Threads: %d\n\n", N, N, threads);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C_ref = (double *)malloc((size_t)N * N * sizeof(double));
    double *C_row = (double *)malloc((size_t)N * N * sizeof(double));
    double *C_col = (double *)malloc((size_t)N * N * sizeof(double));
    double *C_blk = (double *)malloc((size_t)N * N * sizeof(double));

    init_data(A, B, C_ref, N);

    // Reference Row-wise sequential time
    double start_seq = omp_get_wtime();
    matmul_rowwise(A, B, C_ref, N, 1);
    double seq_time = omp_get_wtime() - start_seq;
    printf("Sequential Baseline Time: %.6f s\n\n", seq_time);

    // 1. Row-wise
    init_data(A, B, C_row, N);
    double t0 = omp_get_wtime();
    matmul_rowwise(A, B, C_row, N, threads);
    double t_row = omp_get_wtime() - t0;

    // 2. Column-wise
    init_data(A, B, C_col, N);
    t0 = omp_get_wtime();
    matmul_colwise(A, B, C_col, N, threads);
    double t_col = omp_get_wtime() - t0;

    // 3. Block 2D
    init_data(A, B, C_blk, N);
    t0 = omp_get_wtime();
    matmul_block(A, B, C_blk, N, threads);
    double t_blk = omp_get_wtime() - t0;

    printf("%-20s %-18s %-12s %-12s %-12s\n", "Strategy", "Execution Time (s)", "Speedup", "Efficiency", "Verified");
    printf("------------------------------------------------------------------------\n");
    printf("%-20s %-18.6f %-12.4f %-12.4f %-12s\n", "Row-wise Decomposition", t_row, seq_time / t_row, (seq_time / t_row) / threads, verify_results(C_ref, C_row, N) ? "PASSED" : "FAILED");
    printf("%-20s %-18.6f %-12.4f %-12.4f %-12s\n", "Column-wise Decomp", t_col, seq_time / t_col, (seq_time / t_col) / threads, verify_results(C_ref, C_col, N) ? "PASSED" : "FAILED");
    printf("%-20s %-18.6f %-12.4f %-12.4f %-12s\n", "Block 2D Decomposition", t_blk, seq_time / t_blk, (seq_time / t_blk) / threads, verify_results(C_ref, C_blk, N) ? "PASSED" : "FAILED");

    free(A); free(B); free(C_ref); free(C_row); free(C_col); free(C_blk);
    return 0;
}
