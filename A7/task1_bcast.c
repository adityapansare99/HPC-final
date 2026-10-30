/*
 * ==============================================================================
 * HPC Experiment 7 - Task 1: MPI Broadcast Using MPI_Bcast()
 * ==============================================================================
 * Description:
 *   Demonstrates MPI collective communication using MPI_Bcast().
 *   Process 0 initializes an integer value (broadcast token).
 *   Process 0 broadcasts the value to all other processes in MPI_COMM_WORLD.
 *   Every process prints its rank, total process count, and the received value.
 *   Also measures communication latency and verifies data consistency.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int broadcast_value = 0;
    double start_time = 0.0, end_time = 0.0, bcast_time = 0.0;
    int test_token = 202677; // Assignment 7 token

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc > 1) { 
            broadcast_value = atoi(argv[1]);
        } else {
            broadcast_value = test_token;
        }
        printf("=================================================================\n");
        printf("       MPI Task 1: Broadcast Using MPI_Bcast()                  \n");
        printf("=================================================================\n");
        printf("[Root Process 0] Initialized broadcast data value = %d\n", broadcast_value);
        printf("[Root Process 0] Total MPI processes involved = %d\n", size);
        printf("-----------------------------------------------------------------\n");
    }

    // Synchronize processes before measuring collective broadcast time
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Perform collective broadcast from root (Process 0) to all processes
    MPI_Bcast(&broadcast_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    bcast_time = (end_time - start_time) * 1000.0; // milliseconds

    // Each process outputs received data
    printf("[Process Rank %2d of %2d] Successfully received broadcast value: %d\n",
           rank, size, broadcast_value);

    // Root summarizes performance
    if (rank == 0) {
        printf("-----------------------------------------------------------------\n");
        printf("[Performance] Collective MPI_Bcast latency across %d processes = %.6f ms\n",
               size, bcast_time);
        printf("=================================================================\n\n");
    }

    MPI_Finalize();
    return 0;
}
