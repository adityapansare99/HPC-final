/*
 * ==============================================================================
 * HPC Experiment 7 - Task 4: Parallel MPI Matrix-Vector Multiplication
 * ==============================================================================

 * Description:
 *   Implements distributed Matrix-Vector Multiplication: Y = A * X
 *   where A is (N x N), X is (N x 1), and Y is (N x 1).
 *   - Sequential Baseline on Root Process.
 *   - MPI Parallel Algorithm using Row-Wise Decomposition:
 *     1. MPI_Bcast() to distribute vector X to all processes.
 *     2. MPI_Scatter() to distribute chunks of rows of matrix A.
 *     3. Local Dot-Product computation on assigned matrix rows.
 *     4. MPI_Gather() to collect resulting partial vector Y at Root.
 *   - Measures sequential time, parallel time, speedup, efficiency, and communication overhead.
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
    int N = 1000; // Default matrix dimension
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

    // Calculate non-uniform row distribution among processes
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

    int local_rows = row_counts[rank];
    int local_elements = sendcounts[rank];

    double *A = NULL;
    double *X = (double *)malloc(N * sizeof(double));
    double *Y_seq = NULL;
    double *Y_mpi = NULL;
    double *local_A = (double *)malloc(local_elements * sizeof(double));
    double *local_Y = (double *)malloc(local_rows * sizeof(double));

    double seq_time = 0.0;
    double par_start = 0.0, par_end = 0.0, par_time = 0.0;
    double comp_start = 0.0, comp_end = 0.0, comp_time = 0.0;
    double comm_time = 0.0;

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        Y_seq = (double *)malloc(N * sizeof(double));
        Y_mpi = (double *)malloc(N * sizeof(double));

        // Deterministic initialization
        for (int i = 0; i < N; i++) {
            X[i] = (double)(i % 10 + 1) * 0.5;
            for (int j = 0; j < N; j++) {
                A[i * N + j] = ((i + j) % 20 + 1) * 0.1;
            }
        }

        // Measure Sequential Execution Time
        double s_start = MPI_Wtime();
        sequential_mat_vec(N, A, X, Y_seq);
        double s_end = MPI_Wtime();
        seq_time = (s_end - s_start) * 1000.0; // ms

        if (!benchmark_mode) {
            printf("=================================================================\n");
            printf("       MPI Task 4: Parallel Matrix-Vector Multiplication         \n");
            printf("=================================================================\n");
            printf("[Configuration] Matrix Size: %d x %d | Processes: %d | Base Rows: %d\n",
                   N, N, size, base_rows);
            printf("[Sequential Baseline] Execution Time = %.4f ms\n", seq_time);
            printf("-----------------------------------------------------------------\n");
        }
    }

    // Barrier before parallel timing
    MPI_Barrier(MPI_COMM_WORLD);
    par_start = MPI_Wtime();

    // 1. Broadcast vector X to all processes
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Scatter rows of matrix A using Scatterv for universal compatibility
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, local_elements, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 3. Local computation
    comp_start = MPI_Wtime();
    for (int i = 0; i < local_rows; i++) {
        double sum = 0.0;
        int row_offset = i * N;
        for (int j = 0; j < N; j++) {
            sum += local_A[row_offset + j] * X[j];
        }
        local_Y[i] = sum;
    }
    comp_end = MPI_Wtime();
    comp_time = (comp_end - comp_start) * 1000.0;

    // 4. Gather result vector Y
    MPI_Gatherv(local_Y, local_rows, MPI_DOUBLE,
                Y_mpi, row_counts, row_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    par_end = MPI_Wtime();
    par_time = (par_end - par_start) * 1000.0; // ms

    // Maximum computation time across processes
    double max_comp_time = 0.0;
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        comm_time = par_time - max_comp_time;
        if (comm_time < 0.0) comm_time = 0.0001;

        double speedup = seq_time / par_time;
        double efficiency = (speedup / size) * 100.0;

        // Verify result correctness
        double max_err = 0.0;
        for (int i = 0; i < N; i++) {
            double err = fabs(Y_mpi[i] - Y_seq[i]);
            if (err > max_err) max_err = err;
        }

        if (benchmark_mode) {
            // Machine-readable benchmark line:
            // MATVEC | N | processes | seq_ms | par_ms | comp_ms | comm_ms | speedup | efficiency | max_err
            printf("DATA_MATVEC|%d|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%.2f|%.2e\n",
                   N, size, seq_time, par_time, max_comp_time, comm_time, speedup, efficiency, max_err);
        } else {
            printf("[MPI Parallel Results]\n");
            printf("   - Total Parallel Time  : %.4f ms\n", par_time);
            printf("   - Pure Computation Time: %.4f ms\n", max_comp_time);
            printf("   - Communication Time   : %.4f ms (%.1f%% of total)\n",
                   comm_time, (comm_time / par_time) * 100.0);
            printf("   - Achieved Speedup     : %.4fx\n", speedup);
            printf("   - Parallel Efficiency  : %.2f%%\n", efficiency);
            printf("   - Numerical Max Error  : %.2e (%s)\n",
                   max_err, (max_err < 1e-9) ? "VERIFIED ACCURATE" : "ERROR DETECTED");
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
