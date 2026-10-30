/*
 * ==============================================================================
 * HPC Experiment 7 - Task 7: Non-Uniform Distribution with MPI_Scatterv()
 * ==============================================================================

 * Description:
 *   Demonstrates non-uniform data partitioning where matrix dimension N is NOT
 *   evenly divisible by the process count p (e.g. N = 10, p = 3).
 *   - Dynamically calculates sendcounts[] and displs[] arrays.
 *   - Distributes non-uniform row blocks of matrix A using MPI_Scatterv().
 *   - Broadcasts vector X using MPI_Bcast().
 *   - Computes local dot products on assigned varying row counts.
 *   - Gathers variable-sized result vectors using MPI_Gatherv().
 *   - Verifies against sequential matrix-vector multiplication.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void sequential_mat_vec(int N, const double *A, const double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        int row_offset = i * N;
        for (int j = 0; j < N; j++) {
            sum += A[row_offset + j] * X[j];
        }
        Y[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 10; // Default non-divisible test dimension (e.g. N=10, p=3)
    int benchmark_mode = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        benchmark_mode = atoi(argv[2]);
    }

    // 1. Calculate non-uniform row distribution among processes
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
    int my_elements = sendcounts[rank];

    double *A = NULL;
    double *X = (double *)malloc(N * sizeof(double));
    double *Y_seq = NULL;
    double *Y_mpi = NULL;

    double *local_A = (double *)malloc(my_elements * sizeof(double));
    double *local_Y = (double *)malloc(my_rows * sizeof(double));

    double seq_time = 0.0;
    double par_start, par_end, par_time;

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        Y_seq = (double *)malloc(N * sizeof(double));
        Y_mpi = (double *)malloc(N * sizeof(double));

        // Deterministic initialization
        for (int i = 0; i < N; i++) {
            X[i] = (double)(i % 5 + 1) * 1.0;
            for (int j = 0; j < N; j++) {
                A[i * N + j] = (double)((i + j) % 7 + 1);
            }
        }

        // Sequential calculation
        double s_start = MPI_Wtime();
        sequential_mat_vec(N, A, X, Y_seq);
        double s_end = MPI_Wtime();
        seq_time = (s_end - s_start) * 1000.0; // ms

        if (!benchmark_mode) {
            printf("=================================================================\n");
            printf("  MPI Task 7: Non-Uniform Data Partitioning Using MPI_Scatterv() \n");
            printf("=================================================================\n");
            printf("[Problem Dimension] Matrix Size: %d x %d | Number of Processes: %d\n", N, N, size);
            printf("[Partitioning Scheme] Base Rows = %d, Remainder Rows = %d\n", base_rows, remainder);
            printf("-----------------------------------------------------------------\n");
            printf("Process Rank | Assigned Rows | Row Range    | Elements Received\n");
            printf("-------------+---------------+--------------+------------------\n");
            for (int p = 0; p < size; p++) {
                printf("  Rank %2d    | %7d rows  | [%2d .. %2d]  | %7d doubles\n",
                       p, row_counts[p], row_displs[p], row_displs[p] + row_counts[p] - 1, sendcounts[p]);
            }
            printf("-----------------------------------------------------------------\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    par_start = MPI_Wtime();

    // 1. Broadcast vector X to all processes
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Scatter variable sized row blocks using MPI_Scatterv
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, my_elements, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 3. Compute local matrix-vector multiplication
    for (int i = 0; i < my_rows; i++) {
        double sum = 0.0;
        int r_off = i * N;
        for (int j = 0; j < N; j++) {
            sum += local_A[r_off + j] * X[j];
        }
        local_Y[i] = sum;
    }

    // 4. Gather variable sized results back using MPI_Gatherv
    MPI_Gatherv(local_Y, my_rows, MPI_DOUBLE,
                Y_mpi, row_counts, row_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    par_end = MPI_Wtime();
    par_time = (par_end - par_start) * 1000.0;

    if (!benchmark_mode) {
        printf("[Process Rank %2d] Computed %d rows (offset %d..%d) successfully.\n",
               rank, my_rows, row_displs[rank], row_displs[rank] + my_rows - 1);
    }

    if (rank == 0) {
        // Correctness verification
        double max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double err = fabs(Y_mpi[i] - Y_seq[i]);
            if (err > max_err) max_err = err;
        }

        if (benchmark_mode) {
            printf("DATA_SCATTERV|%d|%d|%.4f|%.4f|%.4f|%.2e\n",
                   N, size, seq_time, par_time, seq_time / par_time, max_err);
        } else {
            printf("-----------------------------------------------------------------\n");
            printf("[Root Verification]\n");
            if (N <= 15) {
                printf("Index | Sequential Y[i] | MPI_Scatterv Y[i] | Status\n");
                printf("------+-----------------+-------------------+--------\n");
                for (int i = 0; i < N; i++) {
                    printf("  %2d  | %15.4f | %17.4f | %s\n",
                           i, Y_seq[i], Y_mpi[i], (fabs(Y_seq[i] - Y_mpi[i]) < 1e-9) ? "MATCH" : "MISMATCH");
                }
            }
            printf("-----------------------------------------------------------------\n");
            printf("Max Numerical Discrepancy : %.2e\n", max_err);
            printf("Verification Status       : %s\n", (max_err < 1e-9) ? "PASSED (PERFECT MATCH)" : "FAILED");
            printf("Sequential Execution Time : %.4f ms\n", seq_time);
            printf("MPI_Scatterv Parallel Time: %.4f ms (Speedup: %.2fx)\n", par_time, seq_time / par_time);
            printf("=================================================================\n\n");
        }

        free(A);
        free(Y_seq);
        free(Y_mpi);
    }

    free(row_counts);
    free(row_displs);
    free(sendcounts);
    free(displs);
    free(X);
    free(local_A);
    free(local_Y);

    MPI_Finalize();
    return 0;
}
