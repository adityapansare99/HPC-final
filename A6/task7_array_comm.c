#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 7: Array Communication & Performance Activity
 * Description: Process 0 transmits an array of integers to Process 1.
 *              Process 1 computes the Sum, Average, and Maximum of the array.
 *              Includes performance timing mode for various message sizes
 *              (1 KB, 10 KB, 100 KB, 1 MB, 10 MB).
 */

void run_basic_demo(int rank) {
    if (rank == 0) {
        int sample_arr[5] = {10, 20, 30, 40, 50};
        int count = 5;

        printf("[Process 0] Initialized array: [ ");
        for (int i = 0; i < count; i++) {
            printf("%d ", sample_arr[i]);
        }
        printf("]\n");

        printf("[Process 0] Sending %d integers to Process 1 via MPI_Send...\n", count);
        MPI_Send(sample_arr, count, MPI_INT, 1, 101, MPI_COMM_WORLD);
        printf("[Process 0] Array transmission complete.\n");
    } else if (rank == 1) {
        int recv_arr[5];
        MPI_Status status;

        MPI_Recv(recv_arr, 5, MPI_INT, 0, 101, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_INT, &count);

        printf("[Process 1] Received array: [ ");
        for (int i = 0; i < count; i++) {
            printf("%d ", recv_arr[i]);
        }
        printf("]\n");

        // Calculations
        long long sum = 0;
        int max_val = recv_arr[0];
        for (int i = 0; i < count; i++) {
            sum += recv_arr[i];
            if (recv_arr[i] > max_val) {
                max_val = recv_arr[i];
            }
        }
        double avg = (double)sum / count;

        printf("-----------------------------------------------------------------\n");
        printf("[Process 1] Array Analytics Results:\n");
        printf("  -> Elements Count : %d\n", count);
        printf("  -> Sum            : %lld\n", sum);
        printf("  -> Average        : %.2f\n", avg);
        printf("  -> Maximum Value  : %d\n", max_val);
        printf("-----------------------------------------------------------------\n");
    }
}

void run_performance_benchmarks(int rank) {
    // Message sizes: 1 KB, 10 KB, 100 KB, 1 MB, 10 MB
    size_t sizes_bytes[] = {
        1024,               // 1 KB
        10 * 1024,          // 10 KB
        100 * 1024,         // 100 KB
        1024 * 1024,        // 1 MB
        10 * 1024 * 1024    // 10 MB
    };
    const char *size_labels[] = {"1 KB", "10 KB", "100 KB", "1 MB", "10 MB"};
    int num_tests = sizeof(sizes_bytes) / sizeof(sizes_bytes[0]);
    int warmup_trials = 5;
    int timed_trials = 20;

    if (rank == 0) {
        printf("\n=================================================================\n");
        printf("       Task 7: Point-to-Point Communication Scaling Benchmark    \n");
        printf("=================================================================\n");
        printf("%-10s | %-12s | %-20s | %-16s\n", "Size Label", "Bytes", "Comm Time (ms)", "Bandwidth (MB/s)");
        printf("-----------------------------------------------------------------\n");
    }

    for (int t = 0; t < num_tests; t++) {
        size_t bytes = sizes_bytes[t];
        int num_ints = (int)(bytes / sizeof(int));
        int *buffer = (int *)malloc(bytes);
        if (!buffer) {
            fprintf(stderr, "Memory allocation failed for size %zu\n", bytes);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank == 0) {
            for (int i = 0; i < num_ints; i++) buffer[i] = i % 1000;
        }

        // Warmup iterations
        for (int w = 0; w < warmup_trials; w++) {
            if (rank == 0) {
                MPI_Send(buffer, num_ints, MPI_INT, 1, 200 + t, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Recv(buffer, num_ints, MPI_INT, 0, 200 + t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Timed benchmark
        double t_start = MPI_Wtime();
        for (int iter = 0; iter < timed_trials; iter++) {
            if (rank == 0) {
                MPI_Send(buffer, num_ints, MPI_INT, 1, 300 + t, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Recv(buffer, num_ints, MPI_INT, 0, 300 + t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        if (rank == 0) {
            double total_time_sec = t_end - t_start;
            double avg_comm_time_ms = (total_time_sec / timed_trials) * 1000.0;
            double mb_transferred = (double)bytes / (1024.0 * 1024.0);
            double bandwidth_mbps = (mb_transferred * timed_trials) / total_time_sec;

            printf("%-10s | %-12zu | %-20.4f | %-16.2f\n",
                   size_labels[t], bytes, avg_comm_time_ms, bandwidth_mbps);
        }

        free(buffer);
    }
    if (rank == 0) {
        printf("=================================================================\n\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Task 7 requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("=================================================================\n");
        printf("       Task 7: Array Communication & Analytics Demonstration    \n");
        printf("=================================================================\n");
    }

    run_basic_demo(rank);
    run_performance_benchmarks(rank);

    MPI_Finalize();
    return 0;
}
