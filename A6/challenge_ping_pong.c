#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/**
 * Assignment 6 - Challenge: Ping-Pong Communication & Round-Trip Time (RTT) Analysis
 * Description: Process 0 transmits a message of size S to Process 1,
 *              Process 1 echoes the message back to Process 0.
 *              This ping-pong loop is executed across N iterations for
 *              different payload sizes (1 KB, 10 KB, 100 KB, 1 MB, 10 MB).
 *              Computes Total Time, Average RTT, One-Way Latency, and Effective Bandwidth.
 */

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Ping-Pong requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Message payload sizes: 1 KB, 10 KB, 100 KB, 1 MB, 10 MB
    size_t sizes_bytes[] = {
        1024,               // 1 KB
        10 * 1024,          // 10 KB
        100 * 1024,         // 100 KB
        1024 * 1024,        // 1 MB
        10 * 1024 * 1024    // 10 MB
    };
    const char *size_labels[] = {"1 KB", "10 KB", "100 KB", "1 MB", "10 MB"};
    int num_sizes = sizeof(sizes_bytes) / sizeof(sizes_bytes[0]);

    // Iteration counts scaled for large sizes to balance execution time & precision
    int iterations[] = {1000, 1000, 500, 100, 20};

    if (rank == 0) {
        printf("========================================================================================\n");
        printf("               Assignment 6 Challenge: Ping-Pong Communication & RTT Analysis            \n");
        printf("========================================================================================\n");
        printf("%-10s | %-10s | %-12s | %-16s | %-14s | %-16s\n",
               "Size Label", "Bytes", "Iterations", "Total Time (s)", "Avg RTT (ms)", "Bandwidth (MB/s)");
        printf("----------------------------------------------------------------------------------------\n");
    }

    for (int i = 0; i < num_sizes; i++) {
        size_t bytes = sizes_bytes[i];
        int iters = iterations[i];
        char *buffer = (char *)malloc(bytes);
        if (!buffer) {
            fprintf(stderr, "Allocation failed for size %zu\n", bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize buffer
        memset(buffer, 'A' + (i % 26), bytes);

        // Warmup (10 iterations)
        for (int w = 0; w < 10; w++) {
            if (rank == 0) {
                MPI_Send(buffer, (int)bytes, MPI_BYTE, 1, 50, MPI_COMM_WORLD);
                MPI_Recv(buffer, (int)bytes, MPI_BYTE, 1, 51, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buffer, (int)bytes, MPI_BYTE, 0, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, (int)bytes, MPI_BYTE, 0, 51, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double t_start = MPI_Wtime();

        for (int iter = 0; iter < iters; iter++) {
            if (rank == 0) {
                MPI_Send(buffer, (int)bytes, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
                MPI_Recv(buffer, (int)bytes, MPI_BYTE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buffer, (int)bytes, MPI_BYTE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, (int)bytes, MPI_BYTE, 0, 101, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        if (rank == 0) {
            double total_time = t_end - t_start;
            double avg_rtt_ms = (total_time / iters) * 1000.0;
            // In each round trip, 2 * bytes are transferred across the network/interconnect
            double total_bytes_transferred = 2.0 * (double)bytes * (double)iters;
            double bandwidth_mbps = (total_bytes_transferred / (1024.0 * 1024.0)) / total_time;

            printf("%-10s | %-10zu | %-12d | %-16.6f | %-14.4f | %-16.2f\n",
                   size_labels[i], bytes, iters, total_time, avg_rtt_ms, bandwidth_mbps);
        }

        free(buffer);
    }

    if (rank == 0) {
        printf("========================================================================================\n");
    }

    MPI_Finalize();
    return 0;
}
