#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void run_sum_experiment(long long N, int num_threads) {
    double *A = (double *)malloc(N * sizeof(double));
    if (!A) {
        printf("Memory allocation failed for N = %lld\n", N);
        return;
    }

    for (long long i = 0; i < N; i++) {
        A[i] = 1.0;
    }

    int repeat = 20; 

    double seq_start = omp_get_wtime();
    double seq_sum = 0.0;
    for (int r = 0; r < repeat; r++) {
        seq_sum = 0.0;
        for (long long i = 0; i < N; i++) {
            seq_sum += A[i];
        }
    }
    double seq_end = omp_get_wtime();
    double seq_time = (seq_end - seq_start) / repeat;
    if (seq_time < 0.0001) seq_time = 0.0001;

    omp_set_num_threads(num_threads);
    double par_start = omp_get_wtime();
    double par_sum = 0.0;

    for (int r = 0; r < repeat; r++) {
        par_sum = 0.0;
        #pragma omp parallel for reduction(+:par_sum)
        for (long long i = 0; i < N; i++) {
            par_sum += A[i];
        }
    }
    double par_end = omp_get_wtime();
    double par_time = (par_end - par_start) / repeat;
    if (par_time < 0.0001) par_time = 0.0001;

    double speedup = seq_time / par_time;

    printf("N = %10lld | Threads = %2d | Seq Time = %.6fs | Par Time = %.6fs | Speedup = %.2fx | Sum = %.0f\n",
           N, num_threads, seq_time, par_time, speedup, par_sum);

    free(A);
}

int main(int argc, char *argv[]) {
    printf("=========================================================\n");
    printf("  Task 2: Parallel Loop Execution (Array Sum) OpenMP      \n");
    printf("=========================================================\n");

    if (argc == 3) {
        long long N = atoll(argv[1]);
        int threads = atoi(argv[2]);
        run_sum_experiment(N, threads);
        return 0;
    }

    long long N_values[] = {10000LL, 1000000LL, 10000000LL};
    int thread_counts[] = {1, 2, 4, 8};

    int num_N = sizeof(N_values) / sizeof(N_values[0]);
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_N; i++) {
        printf("\n--- Experiments for N = %lld ---\n", N_values[i]);
        for (int j = 0; j < num_threads; j++) {
            run_sum_experiment(N_values[i], thread_counts[j]);
        }
    }

    return 0;
}


// gcc -fopenmp -O2 task2_sum.c -o task2.exe ; 
// ./task2.exe