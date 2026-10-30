#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 5: MPI Send and Receive
 * Description: Implements point-to-point communication where Process 0 transmits
 *              a string message to Process 1 using MPI_Send() and MPI_Recv().
 */

#define MAX_STRING_LEN 256

int main(int argc, char *argv[]) {
    int rank, size;
    char message[MAX_STRING_LEN];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Task 5 requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        snprintf(message, MAX_STRING_LEN, "Greetings from Process 0! Message Passing Interface Initialized.");
        printf("[Process 0] Transmitting message: \"%s\"\n", message);
        printf("[Process 0] Destination: Process 1 | Tag: 0 | Communicator: MPI_COMM_WORLD\n");

        // MPI_Send(buf, count, datatype, dest, tag, comm)
        MPI_Send(message, (int)strlen(message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("[Process 0] Message successfully sent and departed sender buffer.\n");
    } else if (rank == 1) {
        // MPI_Recv(buf, count, datatype, source, tag, comm, status)
        MPI_Recv(message, MAX_STRING_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        printf("[Process 1] Successfully received message: \"%s\"\n", message);
        printf("[Process 1] Source: %d | Tag: %d | Characters Received: %d\n",
               status.MPI_SOURCE, status.MPI_TAG, count);
    }

    MPI_Finalize();
    return 0;
}
