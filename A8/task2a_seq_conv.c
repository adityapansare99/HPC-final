#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

void sequential_convolution(int N, const double *input, double *output) {
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

int main(int argc, char *argv[]) {
    int N = 2000;
    int benchmark_mode = 0;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    size_t total_elements = (size_t)N * N;
    double *input = (double *)malloc(total_elements * sizeof(double));
    double *output = (double *)calloc(total_elements, sizeof(double));

    if (!input || !output) {
        fprintf(stderr, "Memory allocation failed for grid %d x %d\n", N, N);
        return 1;
    }

    init_grid(N, input);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sequential_convolution(N, input, output);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    double checksum = 0.0;
    for (int i = 1; i < N - 1; i += 10) {
        for (int j = 1; j < N - 1; j += 10) {
            checksum += output[i * N + j];
        }
    }

    if (benchmark_mode) {
        printf("DATA_SEQ_CONV|%d|1|%.4f|%.4f\n", N, elapsed_ms, checksum);
    } else {
        printf("=================================================================\n");
        printf("        Task 2A: Sequential 2D Convolution on Grid               \n");
        printf("=================================================================\n");
        printf("Grid Dimension    : %d x %d (%zu pixels/cells)\n", N, N, total_elements);
        printf("Kernel Size       : 3 x 3 Spatial Stencil\n");
        printf("Execution Time    : %.4f ms (%.4f seconds)\n", elapsed_ms, elapsed_ms / 1000.0);
        printf("Grid Checksum     : %.4f\n", checksum);
        printf("=================================================================\n");
    }

    free(input);
    free(output);
    return 0;
}
