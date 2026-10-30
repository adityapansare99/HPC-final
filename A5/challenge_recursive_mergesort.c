#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000

void merge(int *arr, int *temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
}

void merge_sort_seq(int *arr, int *temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort_seq(arr, temp, left, mid);
        merge_sort_seq(arr, temp, mid + 1, right);
        merge(arr, temp, left, mid, right);
    }
}

void merge_sort_omp_task(int *arr, int *temp, int left, int right, int threshold) {
    if (left < right) {
        if (right - left < threshold) {
            merge_sort_seq(arr, temp, left, right);
        } else {
            int mid = left + (right - left) / 2;

            #pragma omp task firstprivate(arr, temp, left, mid, threshold)
            merge_sort_omp_task(arr, temp, left, mid, threshold);

            #pragma omp task firstprivate(arr, temp, mid, right, threshold)
            merge_sort_omp_task(arr, temp, mid + 1, right, threshold);

            #pragma omp taskwait
            merge(arr, temp, left, mid, right);
        }
    }
}

void copy_array(int *src, int *dst, int size) {
    for (int i = 0; i < size; i++) dst[i] = src[i];
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Section C Challenge: Recursive Task Parallel Merge Sort \n");
    printf("========================================================================\n");
    printf("Array Size: %d elements\n\n", N);

    int *arr_orig = (int *)malloc(N * sizeof(int));
    int *arr_work = (int *)malloc(N * sizeof(int));
    int *temp = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) arr_orig[i] = rand() % 100000;

    // Baseline Sequential
    copy_array(arr_orig, arr_work, N);
    double t0 = omp_get_wtime();
    merge_sort_seq(arr_work, temp, 0, N - 1);
    double t_seq = omp_get_wtime() - t0;
    printf("Sequential Merge Sort Time: %.6f seconds\n\n", t_seq);

    // Part 1: Thread Scaling Experiment (Threshold = 1000)
    int test_threads[] = {1, 2, 4, 8, omp_get_max_threads()};
    int num_tests = sizeof(test_threads) / sizeof(test_threads[0]);

    printf("--- Thread Scaling Experiment (Threshold = 1000) ---\n");
    printf("%-10s %-20s %-12s %-12s\n", "Threads", "Execution Time (s)", "Speedup", "Efficiency");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < num_tests; i++) {
        int threads = test_threads[i];
        copy_array(arr_orig, arr_work, N);
        omp_set_num_threads(threads);

        t0 = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                merge_sort_omp_task(arr_work, temp, 0, N - 1, 1000);
            }
        }
        double t_par = omp_get_wtime() - t0;
        double speedup = t_seq / t_par;
        double eff = speedup / threads;

        printf("%-10d %-20.6f %-12.2fx %-12.2f%%\n", threads, t_par, speedup, eff * 100.0);
    }

    // Part 2: Granularity Threshold Experiment (Threads = 4)
    int thresholds[] = {10, 100, 1000, 10000};
    int num_thresh = sizeof(thresholds) / sizeof(thresholds[0]);

    printf("\n--- Task Granularity Threshold Experiment (4 Threads) ---\n");
    printf("%-15s %-20s %-12s\n", "Threshold", "Execution Time (s)", "Speedup");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < num_thresh; i++) {
        int thresh = thresholds[i];
        copy_array(arr_orig, arr_work, N);
        omp_set_num_threads(4);

        t0 = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                merge_sort_omp_task(arr_work, temp, 0, N - 1, thresh);
            }
        }
        double t_par = omp_get_wtime() - t0;
        printf("%-15d %-20.6f %-12.2fx\n", thresh, t_par, t_seq / t_par);
    }

    free(arr_orig); free(arr_work); free(temp);
    return 0;
}
