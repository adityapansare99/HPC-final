#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
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

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 2000;
    int benchmark_mode = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) benchmark_mode = atoi(argv[2]);

    int *row_counts = (int *)malloc(size * sizeof(int));
    int *row_displs = (int *)malloc(size * sizeof(int));
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int base_rows = N / size;
    int remainder = N % size;
    int curr_offset = 0;

    for (int p = 0; p < size; p++) {
        row_counts[p] = base_rows + (p < remainder ? 1 : 0);
        row_displs[p] = curr_offset;
        sendcounts[p] = row_counts[p] * N;
        displs[p] = curr_offset * N;
        curr_offset += row_counts[p];
    }

    int my_rows = row_counts[rank];
    size_t my_elements = (size_t)sendcounts[rank];
    size_t total_elements = (size_t)N * N;

    double *global_input = NULL;
    double *global_output = NULL;

    if (rank == 0) {
        global_input = (double *)malloc(total_elements * sizeof(double));
        global_output = (double *)calloc(total_elements, sizeof(double));
        init_grid(N, global_input);
    }

    size_t local_buf_size = (size_t)(my_rows + 2) * N;
    double *local_input_with_halo = (double *)calloc(local_buf_size, sizeof(double));
    double *local_output = (double *)calloc(my_elements, sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Scatterv(global_input, sendcounts, displs, MPI_DOUBLE,
                 &local_input_with_halo[1 * N], (int)my_elements, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double halo_start = MPI_Wtime();
    int top_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Sendrecv(&local_input_with_halo[1 * N], N, MPI_DOUBLE, top_neighbor, 10,
                 &local_input_with_halo[0 * N], N, MPI_DOUBLE, top_neighbor, 20,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&local_input_with_halo[my_rows * N], N, MPI_DOUBLE, bottom_neighbor, 20,
                 &local_input_with_halo[(my_rows + 1) * N], N, MPI_DOUBLE, bottom_neighbor, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double halo_end = MPI_Wtime();
    double halo_ms = (halo_end - halo_start) * 1000.0;

    double comp_start = MPI_Wtime();
    int global_start_row = row_displs[rank];

    for (int r = 1; r <= my_rows; r++) {
        int global_r = global_start_row + (r - 1);
        if (global_r == 0 || global_r == N - 1) continue; 

        int out_r_off = (r - 1) * N;
        for (int c = 1; c < N - 1; c++) {
            double sum = 0.0;
            for (int u = -1; u <= 1; u++) {
                int local_r_off = (r + u) * N;
                for (int v = -1; v <= 1; v++) {
                    sum += local_input_with_halo[local_r_off + (c + v)] * KERNEL[u + 1][v + 1];
                }
            }
            local_output[out_r_off + c] = sum;
        }
    }
    double comp_end = MPI_Wtime();
    double comp_ms = (comp_end - comp_start) * 1000.0;

    MPI_Gatherv(local_output, (int)my_elements, MPI_DOUBLE,
                global_output, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_ms = (end_time - start_time) * 1000.0;

    double max_comp_ms = 0.0, max_halo_ms = 0.0;
    MPI_Reduce(&comp_ms, &max_comp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&halo_ms, &max_halo_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double checksum = 0.0;
        for (int i = 1; i < N - 1; i += 10) {
            for (int j = 1; j < N - 1; j += 10) {
                checksum += global_output[i * N + j];
            }
        }
        double comm_ms = total_ms - max_comp_ms;
        if (comm_ms < 0.0) comm_ms = 0.0001;

        if (benchmark_mode) {
            printf("DATA_MPI_CONV|%d|%d|%.4f|%.4f|%.4f|%.4f|%.4f\n",
                   N, size, total_ms, max_comp_ms, comm_ms, max_halo_ms, checksum);
        } else {
            printf("=================================================================\n");
            printf("    Task 2C: MPI 2D Convolution with Halo / Ghost Rows           \n");
            printf("=================================================================\n");
            printf("Grid Dimension    : %d x %d\n", N, N);
            printf("MPI Processes     : %d\n", size);
            printf("Total Parallel Time: %.4f ms (%.4f seconds)\n", total_ms, total_ms / 1000.0);
            printf("Pure Compute Time : %.4f ms\n", max_comp_ms);
            printf("Halo Exchange Time: %.4f ms\n", max_halo_ms);
            printf("Total Comm Time   : %.4f ms (%.1f%% of total)\n", comm_ms, (comm_ms / total_ms) * 100.0);
            printf("Grid Checksum     : %.4f\n", checksum);
            printf("=================================================================\n");
        }

        free(global_input);
        free(global_output);
    }

    free(row_counts);
    free(row_displs);
    free(sendcounts);
    free(displs);
    free(local_input_with_halo);
    free(local_output);

    MPI_Finalize();
    return 0;
}
