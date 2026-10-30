#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

void matrix_mult_seq(int N, double *A, double *B, double *C) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }
}

void matrix_mult_omp(int N, double *A, double *B, double *C, int num_threads) {
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

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Task 3: Parallel Matrix Multiplication using OpenMP     \n");
    printf("=========================================================\n");

    int sizes[] = {500, 1000, 2000};
    int thread_counts[] = {1, 2, 4, 8};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    if (argc == 3) {
        sizes[0] = atoi(argv[1]);
        num_sizes = 1;
        thread_counts[0] = atoi(argv[2]);
        num_threads = 1;
    }

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("\n---------------------------------------------------------\n");
        printf(" Matrix Size: %d x %d\n", N, N);
        printf("---------------------------------------------------------\n");

        size_t bytes = (size_t)N * N * sizeof(double);
        double *A = (double *)malloc(bytes);
        double *B = (double *)malloc(bytes);
        double *C_seq = (double *)malloc(bytes);
        double *C_par = (double *)malloc(bytes);

        if (!A || !B || !C_seq || !C_par) {
            printf("Memory allocation failed for matrix size %d x %d!\n", N, N);
            if (A) free(A);
            if (B) free(B);
            if (C_seq) free(C_seq);
            if (C_par) free(C_par);
            continue;
        }

        for (int i = 0; i < N * N; i++) {
            A[i] = (double)(i % 100) / 10.0;
            B[i] = (double)((i + 1) % 100) / 10.0;
        }

        double start_seq = omp_get_wtime();
        matrix_mult_seq(N, A, B, C_seq);
        double seq_time = omp_get_wtime() - start_seq;
        if (seq_time < 0.001) seq_time = 0.001; 
        printf("Sequential Execution Time (Ts): %.6f seconds\n\n", seq_time);

        printf("%-10s | %-18s | %-12s | %-12s\n", "Threads(P)", "Execution Time(Tp)", "Speedup(S)", "Efficiency(E)");
        printf("-----------------------------------------------------------------\n");

        for (int t = 0; t < num_threads; t++) {
            int P = thread_counts[t];
            double start_par = omp_get_wtime();
            matrix_mult_omp(N, A, B, C_par, P);
            double par_time = omp_get_wtime() - start_par;
            if (par_time < 0.0001) par_time = 0.0001;

            double speedup = seq_time / par_time;
            double efficiency = speedup / P;

            int correct = 1;
            for (int i = 0; i < N * N; i += N/5 + 1) {
                if (fabs(C_seq[i] - C_par[i]) > 1e-4) {
                    correct = 0;
                    break;
                }
            }

            printf("%-10d | %-18.6f | %-12.4f | %-12.4f %s\n",
                   P, par_time, speedup, efficiency, correct ? "" : "[MISMATCH!]");
        }

        free(A);
        free(B);
        free(C_seq);
        free(C_par);
    }

    return 0;
}

// gcc -fopenmp -O2 task3_matrix_mult.c -o task3.exe ; 
// ./task3.exe