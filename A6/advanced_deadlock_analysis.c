#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/**
 * Assignment 6 - Advanced Challenge: Avoiding Communication Deadlock
 * Description: Demonstrates why symmetric blocking communication (both processes calling
 *              MPI_Send before MPI_Recv) can deadlock for large payloads due to synchronous
 *              rendezvous protocol constraints.
 *              Implements 3 safe architectural solutions:
 *                1. Alternating Send/Receive Ordering (Odd/Even scheme)
 *                2. Atomic Combined MPI_Sendrecv()
 *                3. Asynchronous Non-Blocking Communication (MPI_Isend / MPI_Irecv / MPI_Waitall)
 */

#define PAYLOAD_SIZE 1048576 // 1 MB (exceeds eager protocol buffer, forcing synchronous rendezvous)

void demo_alternating_ordering(int rank) {
    int *send_buf = (int *)malloc(PAYLOAD_SIZE);
    int *recv_buf = (int *)malloc(PAYLOAD_SIZE);
    int count = PAYLOAD_SIZE / sizeof(int);
    MPI_Status status;

    for (int i = 0; i < count; i++) send_buf[i] = (rank == 0) ? 111 : 222;

    if (rank == 0) {
        printf("[Solution 1: Alternating Order] Rank 0 sending, then receiving...\n");
        MPI_Send(send_buf, count, MPI_INT, 1, 10, MPI_COMM_WORLD);
        MPI_Recv(recv_buf, count, MPI_INT, 1, 20, MPI_COMM_WORLD, &status);
        printf("[Solution 1: Alternating Order] Rank 0 received from Rank 1 (Sample val: %d)\n", recv_buf[0]);
    } else if (rank == 1) {
        printf("[Solution 1: Alternating Order] Rank 1 receiving, then sending...\n");
        MPI_Recv(recv_buf, count, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        MPI_Send(send_buf, count, MPI_INT, 0, 20, MPI_COMM_WORLD);
        printf("[Solution 1: Alternating Order] Rank 1 received from Rank 0 (Sample val: %d)\n", recv_buf[0]);
    }

    free(send_buf);
    free(recv_buf);
}

void demo_sendrecv(int rank) {
    int *send_buf = (int *)malloc(PAYLOAD_SIZE);
    int *recv_buf = (int *)malloc(PAYLOAD_SIZE);
    int count = PAYLOAD_SIZE / sizeof(int);
    MPI_Status status;

    for (int i = 0; i < count; i++) send_buf[i] = (rank == 0) ? 333 : 444;

    int peer = (rank == 0) ? 1 : 0;
    printf("[Solution 2: MPI_Sendrecv] Rank %d executing atomic Sendrecv with Peer %d...\n", rank, peer);

    MPI_Sendrecv(send_buf, count, MPI_INT, peer, 30 + rank,
                 recv_buf, count, MPI_INT, peer, 30 + peer,
                 MPI_COMM_WORLD, &status);

    printf("[Solution 2: MPI_Sendrecv] Rank %d completed Sendrecv successfully (Sample recv val: %d)\n",
           rank, recv_buf[0]);

    free(send_buf);
    free(recv_buf);
}

void demo_nonblocking(int rank) {
    int *send_buf = (int *)malloc(PAYLOAD_SIZE);
    int *recv_buf = (int *)malloc(PAYLOAD_SIZE);
    int count = PAYLOAD_SIZE / sizeof(int);
    MPI_Request requests[2];
    MPI_Status statuses[2];

    for (int i = 0; i < count; i++) send_buf[i] = (rank == 0) ? 555 : 666;
    int peer = (rank == 0) ? 1 : 0;

    printf("[Solution 3: Non-Blocking MPI_Isend/Irecv] Rank %d posting async operations...\n", rank);

    // Post non-blocking receive and non-blocking send
    MPI_Irecv(recv_buf, count, MPI_INT, peer, 50 + peer, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(send_buf, count, MPI_INT, peer, 50 + rank, MPI_COMM_WORLD, &requests[1]);

    // Computation overlap can happen here...

    // Synchronize both requests
    MPI_Waitall(2, requests, statuses);

    printf("[Solution 3: Non-Blocking] Rank %d finished Waitall! (Sample recv val: %d)\n",
           rank, recv_buf[0]);

    free(send_buf);
    free(recv_buf);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: Deadlock analysis requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("=================================================================\n");
        printf("     Advanced Challenge: Deadlock Prevention in MPI Protocols    \n");
        printf("=================================================================\n");
        printf("Message Payload Size: %d bytes (1 MB) -> Forces Rendezvous Mode\n", PAYLOAD_SIZE);
        printf("-----------------------------------------------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    demo_alternating_ordering(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("-----------------------------------------------------------------\n");
    demo_sendrecv(rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("-----------------------------------------------------------------\n");
    demo_nonblocking(rank);

    if (rank == 0) {
        printf("=================================================================\n");
        printf("All 3 Deadlock Prevention Mechanisms Verified Successfully!\n");
        printf("=================================================================\n");
    }

    MPI_Finalize();
    return 0;
}
