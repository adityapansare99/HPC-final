#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void init_matrices(int N, double *A, double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((i + 2 * j) % 15 + 1) * 0.1;
            B[i * N + j] = ((2 * i + j) % 15 + 1) * 0.05;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 500;
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

    double *A = NULL;
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C = NULL;

    double *local_A = (double *)malloc(my_elements * sizeof(double));
    double *local_C = (double *)malloc(my_elements * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(total_elements * sizeof(double));
        C = (double *)malloc(total_elements * sizeof(double));
        init_matrices(N, A, B);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, (int)my_elements, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(B, (int)total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double comp_start = MPI_Wtime();
    for (int i = 0; i < my_rows; i++) {
        int i_off = i * N;
        for (int j = 0; j < N; j++) {
            local_C[i_off + j] = 0.0;
        }
        for (int k = 0; k < N; k++) {
            double a_ik = local_A[i_off + k];
            int k_off = k * N;
            for (int j = 0; j < N; j++) {
                local_C[i_off + j] += a_ik * B[k_off + j];
            }
        }
    }
    double comp_end = MPI_Wtime();

    MPI_Gatherv(local_C, (int)my_elements, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_ms = (end_time - start_time) * 1000.0;
    double comp_ms = (comp_end - comp_start) * 1000.0;

    double max_comp_ms = 0.0;
    MPI_Reduce(&comp_ms, &max_comp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double checksum = 0.0;
        for (size_t i = 0; i < total_elements; i += N + 1) checksum += C[i];
        double comm_ms = total_ms - max_comp_ms;
        if (comm_ms < 0.0) comm_ms = 0.0001;

        if (benchmark_mode) {
            printf("DATA_MPI_MATMUL|%d|%d|%.4f|%.4f|%.4f|%.4f\n",
                   N, size, total_ms, max_comp_ms, comm_ms, checksum);
        } else {
            printf("=================================================================\n");
            printf("    Task 1C: MPI Parallel Matrix-Matrix Multiplication           \n");
            printf("=================================================================\n");
            printf("Matrix Dimension  : %d x %d\n", N, N);
            printf("MPI Processes     : %d\n", size);
            printf("Total Parallel Time: %.4f ms (%.4f seconds)\n", total_ms, total_ms / 1000.0);
            printf("Pure Compute Time : %.4f ms\n", max_comp_ms);
            printf("Communication Time: %.4f ms (%.1f%% of total)\n", comm_ms, (comm_ms / total_ms) * 100.0);
            printf("Diagonal Checksum : %.6f\n", checksum);
            printf("=================================================================\n");
        }

        free(A);
        free(C);
    }

    free(row_counts);
    free(row_displs);
    free(sendcounts);
    free(displs);
    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
