#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 3: Understanding MPI Rank and Size
 * Description: Demonstrates process identification and team size discovery
 *              using MPI_Comm_rank() and MPI_Comm_size().
 */

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Retrieve unique process identifier (Rank: 0 to size-1)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Retrieve total number of processes in communicator
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process reports its assigned rank and the overall group size
    printf("Process %d of %d\n", rank, size);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
