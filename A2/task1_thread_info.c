#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

void perform_matrix_addition(int num_threads) {
    omp_set_num_threads(num_threads);

    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    printf("\n--- Executing Parallel Matrix Addition with %d Threads ---\n", num_threads);

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        #pragma omp critical
        {
            printf("[Parallel Region] Thread ID: %d / Total Threads: %d\n", thread_id, total_threads);
        }

        #pragma omp for
        for (int i = 0; i < N * N; i++) {
            C[i] = A[i] + B[i];
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (C[i] != 3.0) {
            correct = 0;
            break;
        }
    }

    printf("Matrix Addition completed in %.6f seconds (Status: %s)\n",
           execution_time, correct ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Task 1: Thread Creation and Identification using OpenMP \n");
    printf("=========================================================\n");

    if (argc > 1) {
        int threads = atoi(argv[1]);
        if (threads > 0) {
            perform_matrix_addition(threads);
            return 0;
        }
    }

    int thread_counts[] = {2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) {
        perform_matrix_addition(thread_counts[i]);
    }

    return 0;
}


// gcc -fopenmp -O2 task1_thread_info.c -o task1.exe ; 
// ./task1.exe
