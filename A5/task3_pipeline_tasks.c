#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 10000000

void stage1_preprocessing(double *data) {
    for (int i = 0; i < N; i++) {
        data[i] = (double)(i % 1000) / 1000.0;
    }
}

void stage2_computation(double *data) {
    for (int i = 0; i < N; i++) {
        double val = data[i];
        data[i] = sin(val) * cos(val) + sqrt(val + 1.0);
    }
}

void stage3_postprocessing(double *data, double *out_sum, double *out_max) {
    double sum = 0.0;
    double max_val = data[0];
    for (int i = 0; i < N; i++) {
        sum += data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    *out_sum = sum;
    *out_max = max_val;
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Task 3: Parallel Pipeline using OpenMP Tasks             \n");
    printf("========================================================================\n");
    printf("Dataset Size: %d elements | 3 Pipeline Stages\n\n", N);

    double *data_seq = (double *)malloc(N * sizeof(double));
    double *data_task = (double *)malloc(N * sizeof(double));

    // Sequential Pipeline Execution
    double sum_seq = 0.0, max_seq = 0.0;
    double t0 = omp_get_wtime();
    stage1_preprocessing(data_seq);
    stage2_computation(data_seq);
    stage3_postprocessing(data_seq, &sum_seq, &max_seq);
    double t_seq = omp_get_wtime() - t0;

    printf("--- Sequential Pipeline Execution ---\n");
    printf("Sum: %.4f | Max: %.4f | Time: %.6f s\n\n", sum_seq, max_seq, t_seq);

    // OpenMP Task-Based Pipeline Execution
    double sum_task = 0.0, max_task = 0.0;
    omp_set_num_threads(4);

    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            stage1_preprocessing(data_task);

            #pragma omp taskwait

            #pragma omp task
            stage2_computation(data_task);

            #pragma omp taskwait

            #pragma omp task
            stage3_postprocessing(data_task, &sum_task, &max_task);

            #pragma omp taskwait
        }
    }
    double t_task = omp_get_wtime() - t0;

    printf("--- OpenMP Task Pipeline Execution (4 Threads) ---\n");
    printf("Sum: %.4f | Max: %.4f | Time: %.6f s\n", sum_task, max_task, t_task);
    printf("Speedup: %.2fx\n", t_seq / t_task);

    free(data_seq);
    free(data_task);
    return 0;
}
