#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 50000000

void init_array(double *arr) {
    for (long i = 0; i < N; i++) {
        arr[i] = (double)((i * 3 + 7) % 1000) / 10.0;
    }
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Task 1: Independent Tasks using OpenMP Sections          \n");
    printf("========================================================================\n");
    printf("Array Size: %d elements\n\n", N);

    double *arr = (double *)malloc((size_t)N * sizeof(double));
    init_array(arr);

    // Part A: Sequential Execution
    double sum_seq = 0.0, max_seq = arr[0], min_seq = arr[0], avg_seq = 0.0;
    double t0 = omp_get_wtime();

    // 1. Sum
    for (long i = 0; i < N; i++) sum_seq += arr[i];
    // 2. Max
    for (long i = 0; i < N; i++) if (arr[i] > max_seq) max_seq = arr[i];
    // 3. Min
    for (long i = 0; i < N; i++) if (arr[i] < min_seq) min_seq = arr[i];
    // 4. Avg
    avg_seq = sum_seq / N;

    double t_seq = omp_get_wtime() - t0;
    printf("--- Part A: Sequential Execution ---\n");
    printf("Sum: %.2f | Max: %.2f | Min: %.2f | Avg: %.2f\n", sum_seq, max_seq, min_seq, avg_seq);
    printf("Sequential Execution Time: %.6f seconds\n\n", t_seq);

    // Part B: OpenMP Parallel Sections
    double sum_par = 0.0, max_par = arr[0], min_par = arr[0], avg_par = 0.0;
    omp_set_num_threads(4);

    t0 = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double local_sum = 0.0;
            for (long i = 0; i < N; i++) local_sum += arr[i];
            sum_par = local_sum;
        }

        #pragma omp section
        {
            double local_max = arr[0];
            for (long i = 0; i < N; i++) if (arr[i] > local_max) local_max = arr[i];
            max_par = local_max;
        }

        #pragma omp section
        {
            double local_min = arr[0];
            for (long i = 0; i < N; i++) if (arr[i] < local_min) local_min = arr[i];
            min_par = local_min;
        }

        #pragma omp section
        {
            double local_sum2 = 0.0;
            for (long i = 0; i < N; i++) local_sum2 += arr[i];
            avg_par = local_sum2 / N;
        }
    }
    double t_sec = omp_get_wtime() - t0;

    printf("--- Part B: OpenMP Sections Execution (4 Threads) ---\n");
    printf("Sum: %.2f | Max: %.2f | Min: %.2f | Avg: %.2f\n", sum_par, max_par, min_par, avg_par);
    printf("OpenMP Sections Execution Time: %.6f seconds\n", t_sec);
    printf("Speedup: %.2fx | Efficiency: %.2f%%\n", t_seq / t_sec, (t_seq / t_sec / 4.0) * 100.0);

    free(arr);
    return 0;
}
