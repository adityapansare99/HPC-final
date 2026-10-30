#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 40000000

void init_array(int *arr) {
    for (int i = 0; i < N; i++) {
        arr[i] = (i * 7 + 13) % 10000;
    }
}

// Implementation 1: Using OpenMP parallel for
void run_parallel_for(int *arr, int threads, long long *out_sum, int *out_max, int *out_min, int *out_even) {
    omp_set_num_threads(threads);
    long long sum = 0;
    int max_val = arr[0];
    int min_val = arr[0];
    int even_count = 0;

    // We execute 4 separate loop passes parallelized via parallel for
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) sum += arr[i];

    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < N; i++) if (arr[i] > max_val) max_val = arr[i];

    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < N; i++) if (arr[i] < min_val) min_val = arr[i];

    #pragma omp parallel for reduction(+:even_count)
    for (int i = 0; i < N; i++) if (arr[i] % 2 == 0) even_count++;

    *out_sum = sum;
    *out_max = max_val;
    *out_min = min_val;
    *out_even = even_count;
}

// Implementation 2: Using OpenMP parallel sections
void run_parallel_sections(int *arr, int threads, long long *out_sum, int *out_max, int *out_min, int *out_even) {
    omp_set_num_threads(threads);
    long long sum = 0;
    int max_val = arr[0];
    int min_val = arr[0];
    int even_count = 0;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            long long s = 0;
            for (int i = 0; i < N; i++) s += arr[i];
            sum = s;
        }

        #pragma omp section
        {
            int mx = arr[0];
            for (int i = 0; i < N; i++) if (arr[i] > mx) mx = arr[i];
            max_val = mx;
        }

        #pragma omp section
        {
            int mn = arr[0];
            for (int i = 0; i < N; i++) if (arr[i] < mn) mn = arr[i];
            min_val = mn;
        }

        #pragma omp section
        {
            int ev = 0;
            for (int i = 0; i < N; i++) if (arr[i] % 2 == 0) ev++;
            even_count = ev;
        }
    }

    *out_sum = sum;
    *out_max = max_val;
    *out_min = min_val;
    *out_even = even_count;
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Task 2: Sections vs Parallel For Comparison             \n");
    printf("========================================================================\n");
    printf("Array Size: %d elements | Threads: 4\n\n", N);

    int *arr = (int *)malloc((size_t)N * sizeof(int));
    init_array(arr);

    long long sum1, sum2;
    int max1, max2, min1, min2, ev1, ev2;

    // Approach 1: Parallel For
    double t0 = omp_get_wtime();
    run_parallel_for(arr, 4, &sum1, &max1, &min1, &ev1);
    double t_for = omp_get_wtime() - t0;

    // Approach 2: Parallel Sections
    t0 = omp_get_wtime();
    run_parallel_sections(arr, 4, &sum2, &max2, &min2, &ev2);
    double t_sec = omp_get_wtime() - t0;

    printf("%-22s %-18s %-12s %-12s %-12s\n", "Approach", "Execution Time (s)", "Sum", "Max", "Even Count");
    printf("------------------------------------------------------------------------\n");
    printf("%-22s %-18.6f %-12lld %-12d %-12d\n", "Parallel For (Loops)", t_for, sum1, max1, ev1);
    printf("%-22s %-18.6f %-12lld %-12d %-12d\n", "Parallel Sections", t_sec, sum2, max2, ev2);

    free(arr);
    return 0;
}
