#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 4: Understanding MPI Groups and Communicators
 * Description: Demonstrates how to extract MPI groups from MPI_COMM_WORLD,
 *              create custom subgroups containing selected ranks (e.g. even ranks 0, 2, 4, 6),
 *              instantiate a new communicator using MPI_Comm_create_group(),
 *              and distinguish between Global Ranks and Sub-communicator Local Ranks.
 */

int main(int argc, char *argv[]) {
    int world_rank, world_size;
    MPI_Group world_group, even_group;
    MPI_Comm even_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Step 1: Extract the base group handle associated with MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Step 2: Determine even ranks to include in the subgroup
    int even_count = (world_size + 1) / 2;
    int *even_ranks = (int *)malloc(even_count * sizeof(int));
    for (int i = 0, r = 0; r < world_size; r += 2, i++) {
        even_ranks[i] = r;
    }

    // Step 3: Create a new group containing only the specified even ranks
    MPI_Group_incl(world_group, even_count, even_ranks, &even_group);

    // Step 4: Create a new communicator from the group
    // Note: MPI_Comm_create_group() is a collective operation across the group members
    // using a unique tag (e.g., 0). Processes not in the group receive MPI_COMM_NULL.
    MPI_Comm_create_group(MPI_COMM_WORLD, even_group, 0, &even_comm);

    if (world_rank == 0) {
        printf("=================================================================\n");
        printf("   Task 4: MPI Groups and Custom Communicators Demonstration    \n");
        printf("=================================================================\n");
        printf("Total World Processes: %d\n", world_size);
        printf("Selected Subgroup: Even Ranks (");
        for (int i = 0; i < even_count; i++) {
            printf("%d%s", even_ranks[i], (i == even_count - 1) ? "" : ", ");
        }
        printf(")\n");
        printf("-----------------------------------------------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Check if the current process is part of the newly created communicator
    if (even_comm != MPI_COMM_NULL) {
        int even_rank, even_size;
        MPI_Comm_rank(even_comm, &even_rank);
        MPI_Comm_size(even_comm, &even_size);

        printf("[MEMBER] Global Rank %d -> Subgroup Local Rank %d (Sub-Comm Size: %d)\n",
               world_rank, even_rank, even_size);

        // Perform a small collective communication (broadcast) inside the new communicator
        int message_val = 0;
        if (even_rank == 0) {
            message_val = 9999;
        }
        MPI_Bcast(&message_val, 1, MPI_INT, 0, even_comm);
        printf("         -> Global Rank %d received broadcast value %d within even_comm.\n",
               world_rank, message_val);

        // Free custom communicator
        MPI_Comm_free(&even_comm);
    } else {
        printf("[NON-MEMBER] Global Rank %d -> Excluded from even_comm (MPI_COMM_NULL)\n", world_rank);
    }

    // Clean up allocated groups and memory
    MPI_Group_free(&world_group);
    MPI_Group_free(&even_group);
    free(even_ranks);

    MPI_Finalize();
    return 0;
}
