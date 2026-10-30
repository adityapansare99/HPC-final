#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * Assignment 6 - Task 1: MPI Installation and Environment Verification
 * Description: Verifies the MPI runtime environment, version specs,
 *              library string, processor name, rank, and total communicator size.
 */

int main(int argc, char *argv[]) {
    int version, subversion;
    int rank, size, len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    char lib_version[MPI_MAX_LIBRARY_VERSION_STRING];
    int lib_len;

    // Initialize the MPI execution environment
    MPI_Init(&argc, &argv);

    // Query MPI standard version and subversion
    MPI_Get_version(&version, &subversion);

    // Query full MPI implementation library version string
    MPI_Get_library_version(lib_version, &lib_len);

    // Query current process rank and total process count in default communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Query host/node processor name
    MPI_Get_processor_name(processor_name, &len);

    if (rank == 0) {
        printf("=================================================================\n");
        printf("       Assignment 6: Task 1 - MPI Environment Verification       \n");
        printf("=================================================================\n");
        printf("MPI Standard Version    : %d.%d\n", version, subversion);
        printf("Total Processes Launched: %d\n", size);
        printf("Master Node Hostname    : %s\n", processor_name);
        printf("MPI Library Version:\n  %s\n", lib_version);
        printf("-----------------------------------------------------------------\n");
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    printf("[Process %d of %d] Active on Node/Core: %s\n", rank, size, processor_name);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

// mpicc task1_env_verify.c -o task
// mpirun -np 4 task