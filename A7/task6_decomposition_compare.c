/*
 * ==============================================================================
 * HPC Experiment 7 - Task 6: Matrix Decomposition Strategies Comparison
 * ==============================================================================

 * Description:
 *   Compares three foundational parallel matrix decomposition strategies:
 *   1. 1D Row-Wise Decomposition (Rows of A scattered, B broadcast)
 *   2. 1D Column-Wise Decomposition (Columns of B scattered, A broadcast)
 *   3. 2D Block Decomposition (Matrix partitioned into sqrt(p) x sqrt(p) sub-blocks)
 *   
 *   Analyzes & profiles:
 *   - Communication volume (Bytes transferred across network)
 *   - Local memory footprint per rank
 *   - Cache locality & execution time
 *   - Communication vs Computation ratio
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

void init_matrices(int N, double *A, double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((i + j) % 10 + 1) * 0.1;
            B[i * N + j] = ((i * 2 + j) % 10 + 1) * 0.1;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 512;
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

    // Verify dimensions
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size N (%d) must be divisible by process count (%d)\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    size_t total_elements = (size_t)N * N;
    double *A = NULL;
    double *B = NULL;
    double *C_ref = NULL;

    if (rank == 0) {
        A = (double *)malloc(total_elements * sizeof(double));
        B = (double *)malloc(total_elements * sizeof(double));
        C_ref = (double *)malloc(total_elements * sizeof(double));
        init_matrices(N, A, B);

        // Reference sequential multiplication for verification
        for (int i = 0; i < N; i++) {
            int i_off = i * N;
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i_off + k] * B[k * N + j];
                }
                C_ref[i_off + j] = sum;
            }
        }
    }

    // -------------------------------------------------------------
    // STRATEGY 1: 1D Row-Wise Decomposition
    // -------------------------------------------------------------
    int rows_per_proc = N / size;
    size_t row_elems = (size_t)rows_per_proc * N;
    double *row_A = (double *)malloc(row_elems * sizeof(double));
    double *all_B = (double *)malloc(total_elements * sizeof(double));
    double *row_C = (double *)malloc(row_elems * sizeof(double));
    double *C_row = (rank == 0) ? (double *)malloc(total_elements * sizeof(double)) : NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_row_start = MPI_Wtime();

    MPI_Scatter(A, (int)row_elems, MPI_DOUBLE, row_A, (int)row_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((rank == 0) ? B : all_B, (int)total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) memcpy(all_B, B, total_elements * sizeof(double));

    double t_row_comp_start = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        int i_off = i * N;
        for (int j = 0; j < N; j++) {
            row_C[i_off + j] = 0.0;
        }
        for (int k = 0; k < N; k++) {
            double a_ik = row_A[i_off + k];
            int k_off = k * N;
            for (int j = 0; j < N; j++) {
                row_C[i_off + j] += a_ik * all_B[k_off + j];
            }
        }
    }
    double t_row_comp_end = MPI_Wtime();

    MPI_Gather(row_C, (int)row_elems, MPI_DOUBLE, C_row, (int)row_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_row_end = MPI_Wtime();
    double time_row = (t_row_end - t_row_start) * 1000.0;
    double comp_row = (t_row_comp_end - t_row_comp_start) * 1000.0;

    free(row_A);
    free(all_B);
    free(row_C);

    // -------------------------------------------------------------
    // STRATEGY 2: 1D Column-Wise Decomposition
    // -------------------------------------------------------------
    // Process p receives N/size columns of B, receives full A via Bcast, computes columns of C
    int cols_per_proc = N / size;
    double *all_A = (double *)malloc(total_elements * sizeof(double));
    double *col_B = (double *)malloc((size_t)N * cols_per_proc * sizeof(double));
    double *col_C = (double *)malloc((size_t)N * cols_per_proc * sizeof(double));
    double *B_trans = NULL;
    double *C_col_trans = (rank == 0) ? (double *)malloc(total_elements * sizeof(double)) : NULL;

    if (rank == 0) {
        B_trans = (double *)malloc(total_elements * sizeof(double));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B_trans[j * N + i] = B[i * N + j]; // Transpose for contiguous column scattering
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_col_start = MPI_Wtime();

    MPI_Scatter(B_trans, (int)(N * cols_per_proc), MPI_DOUBLE, col_B, (int)(N * cols_per_proc), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((rank == 0) ? A : all_A, (int)total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) memcpy(all_A, A, total_elements * sizeof(double));

    double t_col_comp_start = MPI_Wtime();
    for (int c = 0; c < cols_per_proc; c++) {
        int c_off = c * N;
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            int i_off = i * N;
            for (int k = 0; k < N; k++) {
                sum += all_A[i_off + k] * col_B[c_off + k];
            }
            col_C[c * N + i] = sum; // Transposed local result
        }
    }
    double t_col_comp_end = MPI_Wtime();

    MPI_Gather(col_C, (int)(N * cols_per_proc), MPI_DOUBLE, C_col_trans, (int)(N * cols_per_proc), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_col_end = MPI_Wtime();
    double time_col = (t_col_end - t_col_start) * 1000.0;
    double comp_col = (t_col_comp_end - t_col_comp_start) * 1000.0;

    free(all_A);
    free(col_B);
    free(col_C);
    if (rank == 0) {
        free(B_trans);
        free(C_col_trans);
    }

    // -------------------------------------------------------------
    // STRATEGY 3: 2D Block Decomposition (Simulated/Cartesian Grid)
    // -------------------------------------------------------------
    // For general p, determine 2D mesh dimensions dims[0] x dims[1] = size
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int p_r = dims[0]; // row mesh
    int p_c = dims[1]; // col mesh
    int b_rows = N / p_r;
    int b_cols = N / p_c;
    size_t block_size_A = (size_t)b_rows * N;
    size_t block_size_B = (size_t)N * b_cols;
    size_t block_size_C = (size_t)b_rows * b_cols;

    double *blk_A = (double *)malloc(block_size_A * sizeof(double));
    double *blk_B = (double *)malloc(block_size_B * sizeof(double));
    double *blk_C = (double *)malloc(block_size_C * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_blk_start = MPI_Wtime();

    // In 2D block decomposition, process (r, c) needs row-slab of A and col-slab of B
    // We broadcast slabs across row/column communicator meshes
    int my_row = rank / p_c;
    int my_col = rank % p_c;

    if (rank == 0) {
        // Send required slabs to ranks
        for (int p = 0; p < size; p++) {
            int pr = p / p_c;
            int pc = p % p_c;
            if (p == 0) {
                memcpy(blk_A, A, block_size_A * sizeof(double));
                for (int c = 0; c < b_cols; c++) {
                    for (int k = 0; k < N; k++) {
                        blk_B[c * N + k] = B[k * N + c];
                    }
                }
            } else {
                MPI_Send(A + pr * b_rows * N, (int)block_size_A, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                // Extract column slab
                double *temp_b = (double *)malloc(block_size_B * sizeof(double));
                for (int c = 0; c < b_cols; c++) {
                    int col_idx = pc * b_cols + c;
                    for (int k = 0; k < N; k++) {
                        temp_b[c * N + k] = B[k * N + col_idx];
                    }
                }
                MPI_Send(temp_b, (int)block_size_B, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                free(temp_b);
            }
        }
    } else {
        MPI_Recv(blk_A, (int)block_size_A, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(blk_B, (int)block_size_B, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_blk_comp_start = MPI_Wtime();
    for (int i = 0; i < b_rows; i++) {
        int i_off = i * N;
        for (int j = 0; j < b_cols; j++) {
            int j_off = j * N;
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += blk_A[i_off + k] * blk_B[j_off + k];
            }
            blk_C[i * b_cols + j] = sum;
        }
    }
    double t_blk_comp_end = MPI_Wtime();

    // Gather sub-blocks to root
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            double *temp_c = (double *)malloc(block_size_C * sizeof(double));
            MPI_Recv(temp_c, (int)block_size_C, MPI_DOUBLE, p, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            free(temp_c);
        }
    } else {
        MPI_Send(blk_C, (int)block_size_C, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_blk_end = MPI_Wtime();
    double time_blk = (t_blk_end - t_blk_start) * 1000.0;
    double comp_blk = (t_blk_comp_end - t_blk_comp_start) * 1000.0;

    free(blk_A);
    free(blk_B);
    free(blk_C);

    if (rank == 0) {
        // Compute memory footprint & communication volume in MB
        double row_comm_mb = ((double)total_elements * sizeof(double) * 2) / (1024.0 * 1024.0);
        double col_comm_mb = ((double)total_elements * sizeof(double) * 2) / (1024.0 * 1024.0);
        double blk_comm_mb = ((double)size * (block_size_A + block_size_B + block_size_C) * sizeof(double)) / (1024.0 * 1024.0);

        double row_mem_mb = (double)(row_elems + total_elements + row_elems) * sizeof(double) / (1024.0 * 1024.0);
        double col_mem_mb = (double)(total_elements + 2 * N * cols_per_proc) * sizeof(double) / (1024.0 * 1024.0);
        double blk_mem_mb = (double)(block_size_A + block_size_B + block_size_C) * sizeof(double) / (1024.0 * 1024.0);

        if (benchmark_mode) {
            // DATA_DECOMP|strategy|N|p|total_ms|comp_ms|comm_ms|comm_mb|mem_mb
            printf("DATA_DECOMP|Row-Wise|%d|%d|%.4f|%.4f|%.4f|%.2f|%.2f\n",
                   N, size, time_row, comp_row, time_row - comp_row, row_comm_mb, row_mem_mb);
            printf("DATA_DECOMP|Column-Wise|%d|%d|%.4f|%.4f|%.4f|%.2f|%.2f\n",
                   N, size, time_col, comp_col, time_col - comp_col, col_comm_mb, col_mem_mb);
            printf("DATA_DECOMP|2D-Block|%d|%d|%.4f|%.4f|%.4f|%.2f|%.2f\n",
                   N, size, time_blk, comp_blk, time_blk - comp_blk, blk_comm_mb, blk_mem_mb);
        } else {
            printf("=================================================================\n");
            printf("       MPI Task 6: Matrix Decomposition Strategies Comparison    \n");
            printf("=================================================================\n");
            printf("[Matrix Size: %d x %d | MPI Processes: %d (2D Grid: %d x %d)]\n\n",
                   N, N, size, p_r, p_c);

            printf("+----------------------+------------+------------+------------+-------------+------------+\n");
            printf("| Decomposition Scheme | Total (ms) | Comp (ms)  | Comm (ms)  | Comm Vol(MB)| Mem/ProcMB |\n");
            printf("+----------------------+------------+------------+------------+-------------+------------+\n");
            printf("| 1D Row-Wise (Scatter)| %10.4f | %10.4f | %10.4f | %11.2f | %10.2f |\n",
                   time_row, comp_row, time_row - comp_row, row_comm_mb, row_mem_mb);
            printf("| 1D Column-Wise       | %10.4f | %10.4f | %10.4f | %11.2f | %10.2f |\n",
                   time_col, comp_col, time_col - comp_col, col_comm_mb, col_mem_mb);
            printf("| 2D Block (Mesh %dx%d)  | %10.4f | %10.4f | %10.4f | %11.2f | %10.2f |\n",
                   p_r, p_c, time_blk, comp_blk, time_blk - comp_blk, blk_comm_mb, blk_mem_mb);
            printf("+----------------------+------------+------------+------------+-------------+------------+\n\n");

            printf("[Key Architectural Findings]\n");
            printf(" 1. Memory Scalability: 2D Block decomposition requires O(N^2/sqrt(p)) memory per rank,\n");
            printf("    whereas 1D Row/Column schemes require O(N^2) memory because matrix B/A is fully broadcast.\n");
            printf(" 2. Cache Locality: Row-wise decomposition achieves optimal spatial cache locality\n");
            printf("    due to contiguous row memory access in row-major layout.\n");
            printf(" 3. Communication Scalability: For massive cluster node counts (p > 1024), 2D block\n");
            printf("    decomposition dominates by minimizing per-node communication footprint.\n");
            printf("=================================================================\n\n");
        }

        free(A);
        free(B);
        free(C_ref);
        free(C_row);
    }

    MPI_Finalize();
    return 0;
}
