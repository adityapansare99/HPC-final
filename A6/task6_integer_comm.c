#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 6: Integer Data Communication
 * Description: Process 0 transmits a specific integer value (e.g. 100) to Process 1.
 *              Process 1 receives, verifies, and asserts data integrity.
 */

int main(int argc, char *argv[]) {
    int rank, size;
    int send_val = 100;
    int recv_val = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Task 6 requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        if (argc > 1) {
            send_val = atoi(argv[1]);
        }
        printf("[Process 0] Value prepared: %d\n", send_val);
        printf("[Process 0] Sending integer to Process 1 using MPI_INT...\n");

        MPI_Send(&send_val, 1, MPI_INT, 1, 42, MPI_COMM_WORLD);
        printf("[Process 0] Send completed.\n");
    } else if (rank == 1) {
        printf("[Process 1] Waiting to receive integer from Process 0...\n");
        MPI_Recv(&recv_val, 1, MPI_INT, 0, 42, MPI_COMM_WORLD, &status);

        printf("[Process 1] Received Value = %d\n", recv_val);
        printf("[Process 1] Verification: %s (Status Error Code: %d)\n",
               (recv_val == 100 || (argc > 1 && recv_val == atoi(argv[1]))) ? "SUCCESS (Data Matches)" : "FAILURE",
               status.MPI_ERROR);
    }

    MPI_Finalize();
    return 0;
}
