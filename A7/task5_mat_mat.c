/*
 * ==============================================================================
 * HPC Experiment 7 - Task 5: Parallel MPI Matrix-Matrix Multiplication
 * ==============================================================================

 * Description:
 *   Implements distributed Matrix-Matrix Multiplication: C = A * B
 *   where A, B, and C are (N x N) square dense matrices.
 *   - Sequential Baseline on Process 0.
 *   - MPI Parallel Algorithm using Row-Wise Decomposition:
 *     1. MPI_Scatter() distributes (N / p) rows of matrix A to each process.
 *     2. MPI_Bcast() broadcasts the entire matrix B to all processes.
 *     3. Each process computes its local rows of matrix C with cache-friendly loop order.
 *     4. MPI_Gather() collects the resulting rows into C at Process 0.
 *   - Detailed profiling of compute time, scatter time, broadcast time, gather time.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void sequential_mat_mat(int N, const double *A, const double *B, double *C) {
    // Cache-friendly i-k-j loop ordering
    for (int i = 0; i < N; i++) {
        int i_off = i * N;
        for (int j = 0; j < N; j++) {
            C[i_off + j] = 0.0;
        }
        for (int k = 0; k < N; k++) {
            double a_ik = A[i_off + k];
            int k_off = k * N;
            for (int j = 0; j < N; j++) {
                C[i_off + j] += a_ik * B[k_off + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 512; // Default matrix dimension
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
    size_t total_elements = (size_t)N * N;
    size_t local_elements = (size_t)sendcounts[rank];

    double *A = NULL;
    double *B = (double *)malloc(total_elements * sizeof(double));
    double *C_seq = NULL;
    double *C_mpi = NULL;

    double *local_A = (double *)malloc(local_elements * sizeof(double));
    double *local_C = (double *)malloc(local_elements * sizeof(double));

    double seq_time = 0.0;
    double par_start = 0.0, par_end = 0.0, par_time = 0.0;
    double comp_start = 0.0, comp_end = 0.0, comp_time = 0.0;

    if (rank == 0) {
        A = (double *)malloc(total_elements * sizeof(double));
        C_seq = (double *)malloc(total_elements * sizeof(double));
        C_mpi = (double *)malloc(total_elements * sizeof(double));

        // Deterministic matrix initialization
        for (size_t i = 0; i < (size_t)N; i++) {
            for (size_t j = 0; j < (size_t)N; j++) {
                A[i * N + j] = ((i + 2 * j) % 15 + 1) * 0.1;
                B[i * N + j] = ((2 * i + j) % 15 + 1) * 0.05;
            }
        }

        // Sequential calculation & timing
        double s_start = MPI_Wtime();
        sequential_mat_mat(N, A, B, C_seq);
        double s_end = MPI_Wtime();
        seq_time = (s_end - s_start) * 1000.0; // ms

        if (!benchmark_mode) {
            printf("=================================================================\n");
            printf("       MPI Task 5: Parallel Matrix-Matrix Multiplication         \n");
            printf("=================================================================\n");
            printf("[Configuration] Matrix Dimension: %d x %d | Total Elements: %zu\n", N, N, total_elements);
            printf("[Configuration] MPI Processes: %d | Base Rows/Proc: %d\n", size, base_rows);
            printf("[Sequential Baseline] Execution Time = %.4f ms (%.4f seconds)\n",
                   seq_time, seq_time / 1000.0);
            printf("-----------------------------------------------------------------\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    par_start = MPI_Wtime();

    // 1. Scatter rows of A
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, (int)local_elements, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 2. Broadcast entire matrix B to all processes
    MPI_Bcast(B, (int)total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Local matrix multiplication: local_C = local_A * B
    comp_start = MPI_Wtime();
    for (int i = 0; i < local_rows; i++) {
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
    comp_end = MPI_Wtime();
    comp_time = (comp_end - comp_start) * 1000.0;

    // 4. Gather computed rows of C back to Root
    MPI_Gatherv(local_C, (int)local_elements, MPI_DOUBLE,
                C_mpi, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    par_end = MPI_Wtime();
    par_time = (par_end - par_start) * 1000.0; // ms

    // Aggregate maximum compute time across ranks
    double max_comp_time = 0.0;
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double comm_time = par_time - max_comp_time;
        if (comm_time < 0.0) comm_time = 0.0001;
        double speedup = seq_time / par_time;
        double efficiency = (speedup / size) * 100.0;

        // Verify result correctness
        double max_err = 0.0;
        for (size_t i = 0; i < total_elements; i++) {
            double err = fabs(C_mpi[i] - C_seq[i]);
            if (err > max_err) max_err = err;
        }

        if (benchmark_mode) {
            // DATA_MATMAT|N|processes|seq_ms|par_ms|comp_ms|comm_ms|speedup|efficiency|max_err
            printf("DATA_MATMAT|%d|%d|%.4f|%.4f|%.4f|%.4f|%.4f|%.2f|%.2e\n",
                   N, size, seq_time, par_time, max_comp_time, comm_time, speedup, efficiency, max_err);
        } else {
            printf("[MPI Parallel Results]\n");
            printf("   - Total Parallel Time  : %.4f ms (%.4f s)\n", par_time, par_time / 1000.0);
            printf("   - Pure Computation Time: %.4f ms (%.4f s)\n", max_comp_time, max_comp_time / 1000.0);
            printf("   - Communication Time   : %.4f ms (%.1f%% of total)\n",
                   comm_time, (comm_time / par_time) * 100.0);
            printf("   - Achieved Speedup     : %.4fx\n", speedup);
            printf("   - Parallel Efficiency  : %.2f%%\n", efficiency);
            printf("   - Numerical Max Error  : %.2e (%s)\n",
                   max_err, (max_err < 1e-8) ? "VERIFIED ACCURATE" : "ERROR DETECTED");
            printf("=================================================================\n\n");
        }

        free(A);
        free(C_seq);
        free(C_mpi);
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
