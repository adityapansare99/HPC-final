#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000
#define THRESHOLD 10000

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

void merge_sort_omp_tasks(int *arr, int *temp, int left, int right) {
    if (left < right) {
        if (right - left < THRESHOLD) {
            merge_sort_seq(arr, temp, left, right);
        } else {
            int mid = left + (right - left) / 2;
            #pragma omp task firstprivate(arr, temp, left, mid)
            merge_sort_omp_tasks(arr, temp, left, mid);

            #pragma omp task firstprivate(arr, temp, mid, right)
            merge_sort_omp_tasks(arr, temp, mid + 1, right);

            #pragma omp taskwait
            merge(arr, temp, left, mid, right);
        }
    }
}

int is_sorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 4 Task 5: Section C Challenge - Recursive Task Parallelism \n");
    printf("========================================================================\n");
    printf("Array Size: %d | Task Threshold: %d\n\n", N, THRESHOLD);

    int *arr_seq = (int *)malloc(N * sizeof(int));
    int *arr_par = (int *)malloc(N * sizeof(int));
    int *temp_seq = (int *)malloc(N * sizeof(int));
    int *temp_par = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        int val = rand() % 10000;
        arr_seq[i] = val;
        arr_par[i] = val;
    }

    // Sequential Merge Sort
    double t0 = omp_get_wtime();
    merge_sort_seq(arr_seq, temp_seq, 0, N - 1);
    double t_seq = omp_get_wtime() - t0;

    // Parallel OpenMP Task Merge Sort
    omp_set_num_threads(4);
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            merge_sort_omp_tasks(arr_par, temp_par, 0, N - 1);
        }
    }
    double t_par = omp_get_wtime() - t0;

    printf("Sequential Time: %.6f s\n", t_seq);
    printf("Parallel OpenMP Task Time: %.6f s | Speedup: %.2fx | Sorted: %s\n",
           t_par, t_seq / t_par, is_sorted(arr_par, N) ? "YES" : "NO");

    free(arr_seq); free(arr_par); free(temp_seq); free(temp_par);
    return 0;
}
