#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void run_reduction_experiment(long N, int num_threads) {
    double *arr = (double *)malloc(N * sizeof(double));
    if (!arr) {
        printf("Memory allocation failed for N = %ld\n", N);
        return;
    }

    for (long i = 0; i < N; i++) {
        arr[i] = 1.0;
    }

    // 1. Sequential Implementation
    double start_seq = omp_get_wtime();
    double sum_seq = 0.0;
    for (long i = 0; i < N; i++) {
        sum_seq += arr[i];
    }
    double time_seq = omp_get_wtime() - start_seq;
    if (time_seq < 1e-6) time_seq = 1e-6;

    // 2. OpenMP using Critical
    omp_set_num_threads(num_threads);
    double start_crit = omp_get_wtime();
    double sum_crit = 0.0;
    #pragma omp parallel
    {
        double local_sum = 0.0;
        #pragma omp for
        for (long i = 0; i < N; i++) {
            local_sum += arr[i];
        }
        #pragma omp critical
        {
            sum_crit += local_sum;
        }
    }
    double time_crit = omp_get_wtime() - start_crit;
    if (time_crit < 1e-6) time_crit = 1e-6;

    // 3. OpenMP using Reduction
    double start_red = omp_get_wtime();
    double sum_red = 0.0;
    #pragma omp parallel for reduction(+:sum_red)
    for (long i = 0; i < N; i++) {
        sum_red += arr[i];
    }
    double time_red = omp_get_wtime() - start_red;
    if (time_red < 1e-6) time_red = 1e-6;

    printf("\n--- Results for N = %ld (Threads = %d) ---\n", N, num_threads);
    printf("Sequential Time: %.6f s | Sum: %.1f\n", time_seq, sum_seq);
    printf("Critical Time  : %.6f s | Speedup: %.2fx | Sum: %.1f\n", time_crit, time_seq / time_crit, sum_crit);
    printf("Reduction Time : %.6f s | Speedup: %.2fx | Sum: %.1f\n", time_red, time_seq / time_red, sum_red);

    free(arr);
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Assignment 3 - Task 1: Array Sum Reduction Comparison \n");
    printf("=========================================================\n");

    long test_sizes[] = {10000L, 1000000L, 10000000L};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        run_reduction_experiment(test_sizes[i], 4);
    }

    return 0;
}
