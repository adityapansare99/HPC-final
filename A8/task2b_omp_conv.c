#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

static const double KERNEL[3][3] = {
    { 1.0/16.0, 2.0/16.0, 1.0/16.0 },
    { 2.0/16.0, 4.0/16.0, 2.0/16.0 },
    { 1.0/16.0, 2.0/16.0, 1.0/16.0 }
};

void init_grid(int N, double *grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i * N + j] = ((i * 3 + j * 7) % 256) * 1.0;
        }
    }
}

void openmp_conv(int N, const double *input, double *output, int num_threads, int sched_mode) {
    if (sched_mode == 1) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
        for (int i = 1; i < N - 1; i++) {
            int i_off = i * N;
            for (int j = 1; j < N - 1; j++) {
                double sum = 0.0;
                for (int u = -1; u <= 1; u++) {
                    int row_off = (i + u) * N;
                    for (int v = -1; v <= 1; v++) {
                        sum += input[row_off + (j + v)] * KERNEL[u + 1][v + 1];
                    }
                }
                output[i_off + j] = sum;
            }
        }
    } else if (sched_mode == 2) {
        #pragma omp parallel for num_threads(num_threads) schedule(guided, 16)
        for (int i = 1; i < N - 1; i++) {
            int i_off = i * N;
            for (int j = 1; j < N - 1; j++) {
                double sum = 0.0;
                for (int u = -1; u <= 1; u++) {
                    int row_off = (i + u) * N;
                    for (int v = -1; v <= 1; v++) {
                        sum += input[row_off + (j + v)] * KERNEL[u + 1][v + 1];
                    }
                }
                output[i_off + j] = sum;
            }
        }
    } else {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            int i_off = i * N;
            for (int j = 1; j < N - 1; j++) {
                double sum = 0.0;
                for (int u = -1; u <= 1; u++) {
                    int row_off = (i + u) * N;
                    for (int v = -1; v <= 1; v++) {
                        sum += input[row_off + (j + v)] * KERNEL[u + 1][v + 1];
                    }
                }
                output[i_off + j] = sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 2000;
    int num_threads = 4;
    int sched_mode = 0; 
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    if (argc > 3) sched_mode = atoi(argv[3]);
    if (argc > 4) benchmark_mode = atoi(argv[4]);

    if (num_threads <= 0) num_threads = omp_get_max_threads();

    const char *sched_name = (sched_mode == 1) ? "dynamic" : (sched_mode == 2) ? "guided" : "static";

    size_t total_elements = (size_t)N * N;
    double *input = (double *)malloc(total_elements * sizeof(double));
    double *output = (double *)calloc(total_elements, sizeof(double));

    if (!input || !output) {
        fprintf(stderr, "Memory allocation failed for grid %d x %d\n", N, N);
        return 1;
    }

    init_grid(N, input);

    double start_time = omp_get_wtime();
    openmp_conv(N, input, output, num_threads, sched_mode);
    double end_time = omp_get_wtime();
    double elapsed_ms = (end_time - start_time) * 1000.0;

    double checksum = 0.0;
    for (int i = 1; i < N - 1; i += 10) {
        for (int j = 1; j < N - 1; j += 10) {
            checksum += output[i * N + j];
        }
    }

    if (benchmark_mode) {
        printf("DATA_OMP_CONV|%d|%d|%s|%.4f|%.4f\n", N, num_threads, sched_name, elapsed_ms, checksum);
    } else {
        printf("=================================================================\n");
        printf("    Task 2B: OpenMP 2D Convolution & Scheduling Exploration      \n");
        printf("=================================================================\n");
        printf("Grid Dimension    : %d x %d\n", N, N);
        printf("OpenMP Threads    : %d (Max System Threads: %d)\n", num_threads, omp_get_max_threads());
        printf("Schedule Strategy : %s\n", sched_name);
        printf("Execution Time    : %.4f ms (%.4f seconds)\n", elapsed_ms, elapsed_ms / 1000.0);
        printf("Grid Checksum     : %.4f\n", checksum);
        printf("=================================================================\n");
    }

    free(input);
    free(output);
    return 0;
}
