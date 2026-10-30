#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000
#define OUTER_REPS 50000

void task2_array_squares(int num_threads) {
    long long results[N];
    omp_set_num_threads(num_threads);

    printf("\n>>> Task 2 Execution with %d Threads <<<\n", num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        #pragma omp critical
        {
            printf("  [Parallel Region] Thread ID: %2d | Total Threads: %2d\n", tid, total_threads);
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            results[i] = (long long)i * i;
        }
    }
}

double benchmark_squares(int num_threads) {
    long long results[N];
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();

    if (num_threads == 1) {
        for (int rep = 0; rep < OUTER_REPS; rep++) {
            long long sum = 0;
            for (int i = 0; i < N; i++) {
                long long sq = (long long)i * i;
                results[i] = sq;
                sum += sq;
            }
        }
    } else {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int rep = 0; rep < OUTER_REPS; rep++) {
                long long sum = 0;
                for (int i = 0; i < N; i++) {
                    long long sq = (long long)i * i;
                    results[i] = sq;
                    sum += sq;
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    printf("=======================================================================\n");
    printf("   Task 2: OpenMP Thread Creation & Square Calculation (0..%d)\n", N - 1);
    printf("=======================================================================\n");

    printf("\nVerifying square calculation for first 10 whole numbers (0..9):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d^2 = %lld\n", i, (long long)i * i);
    }
    printf("  ...\n");
    printf("  999^2 = %lld\n", (long long)999 * 999);

    int thread_counts[] = {2, 4, 8, 16};
    int num_runs = 4;

    printf("\n-----------------------------------------------------------------------\n");
    printf(" Demonstrating OpenMP Parallel Region & Thread Creation Info           \n");
    printf("-----------------------------------------------------------------------\n");
    for (int i = 0; i < num_runs; i++) {
        task2_array_squares(thread_counts[i]);
    }

    printf("\n=======================================================================\n");
    printf("                 BENCHMARK & SPEEDUP EVALUATION                         \n");
    printf("=======================================================================\n");

    double seq_time = benchmark_squares(1);
    printf("Sequential Execution Time (1 Thread): %f seconds\n\n", seq_time);

    printf("%-20s | %-20s | %-20s\n", "Execution Mode", "Execution Time (s)", "Speedup (Tseq / Tpar)");
    printf("-----------------------------------------------------------------------\n");
    printf("%-20s | %-20.6f | %-20.4f\n", "Sequential (1)", seq_time, 1.0);

    for (int i = 0; i < num_runs; i++) {
        int threads = thread_counts[i];
        double par_time = benchmark_squares(threads);
        double speedup = seq_time / par_time;
        char mode_str[35];
        sprintf(mode_str, "OpenMP (%d Threads)", threads);
        printf("%-20s | %-20.6f | %-20.4f\n", mode_str, par_time, speedup);
    }
    printf("=======================================================================\n");

    return 0;
}


// gcc -fopenmp -O2 task2_squares.c -o task2_squares
// ./task2_squares