#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void benchmark_dot_product(long N, int P) {
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    if (!A || !B) {
        printf("Memory allocation failed!\n");
        return;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.5;
        B[i] = 2.0;
    }

    omp_set_num_threads(P);

    // 1. Sequential
    double start = omp_get_wtime();
    double dot_seq = 0.0;
    for (long i = 0; i < N; i++) {
        dot_seq += A[i] * B[i];
    }
    double time_seq = omp_get_wtime() - start;

    // 2. OpenMP Reduction Clause
    start = omp_get_wtime();
    double dot_red = 0.0;
    #pragma omp parallel for reduction(+:dot_red)
    for (long i = 0; i < N; i++) {
        dot_red += A[i] * B[i];
    }
    double time_red = omp_get_wtime() - start;

    // 3. OpenMP Critical Construct
    start = omp_get_wtime();
    double dot_crit = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (long i = 0; i < N; i++) {
            local_sum += A[i] * B[i];
        }
        #pragma omp critical
        {
            dot_crit += local_sum;
        }
    }
    double time_crit = omp_get_wtime() - start;

    // 4. OpenMP Atomic Directive
    start = omp_get_wtime();
    double dot_atom = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (long i = 0; i < N; i++) {
            local_sum += A[i] * B[i];
        }
        #pragma omp atomic
        dot_atom += local_sum;
    }
    double time_atom = omp_get_wtime() - start;
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Assignment 3 - Task 2 & Section C: Dot Product Analysis\n");
    printf("=========================================================\n");

    long N = 10000000L;
    int thread_counts[] = {1, 2, 4, 8};
    int num_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_counts; i++) {
        benchmark_dot_product(N, thread_counts[i]);
    }

    return 0;
}
