/*
 * ==============================================================================
 * HPC Experiment 7 - Task 2: MPI Scatter and Gather
 * ==============================================================================

 * Description:
 *   Demonstrates MPI collective data distribution and collection.
 *   - Process 0 initializes an integer array of size N (e.g. 16 elements: 1..16).
 *   - Uses MPI_Scatter() to distribute contiguous equal chunks to all processes.
 *   - Each process computes the sum of its received sub-array elements.
 *   - Each process transmits its local sum to Process 0 using MPI_Gather().
 *   - Process 0 aggregates the gathered sums and verifies against theoretical sum N*(N+1)/2.
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEFAULT_ARRAY_SIZE 16

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size = DEFAULT_ARRAY_SIZE;
    int *global_array = NULL;
    int *local_array = NULL;
    int *gathered_sums = NULL;
    int chunk_size = 0;
    long long local_sum = 0;
    long long total_sum = 0;
    double start_time, end_time, scatter_gather_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        array_size = atoi(argv[1]);
    }

    if (array_size % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Array size (%d) must be evenly divisible by number of processes (%d).\n",
                    array_size, size);
        }
        MPI_Finalize();
        return 1;
    }

    chunk_size = array_size / size;
    local_array = (int *)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        global_array = (int *)malloc(array_size * sizeof(int));
        gathered_sums = (int *)malloc(size * sizeof(int));

        printf("=================================================================\n");
        printf("       MPI Task 2: Array Distribution via Scatter and Gather     \n");
        printf("=================================================================\n");
        printf("[Root Process 0] Initializing Array of Size %d across %d processes\n",
               array_size, size);
        printf("[Root Process 0] Chunk size per process = %d elements\n", chunk_size);
        printf("[Root Process 0] Global Array: [");
        for (int i = 0; i < array_size; i++) {
            global_array[i] = i + 1; // 1, 2, 3, ..., 16
            if (i < 16) {
                printf("%d%s", global_array[i], (i == array_size - 1 || i == 15) ? "" : ", ");
            }
        }
        if (array_size > 16) printf(", ... (%d elements)", array_size);
        printf("]\n");
        printf("-----------------------------------------------------------------\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // 1. Scatter the global array from Process 0 to all processes
    MPI_Scatter(global_array, chunk_size, MPI_INT,
                local_array, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // 2. Each process calculates local sum
    local_sum = 0;
    for (int i = 0; i < chunk_size; i++) {
        local_sum += local_array[i];
    }

    // Print local subarray and sum
    printf("[Process Rank %2d] Received chunk of %d elements: [", rank, chunk_size);
    for (int i = 0; i < chunk_size && i < 10; i++) {
        printf("%d%s", local_array[i], (i == chunk_size - 1) ? "" : ", ");
    }
    if (chunk_size > 10) printf(", ...");
    printf("] -> Local Sum = %lld\n", local_sum);

    // 3. Gather local sums at Process 0
    int local_sum_int = (int)local_sum;
    MPI_Gather(&local_sum_int, 1, MPI_INT,
               gathered_sums, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    scatter_gather_time = (end_time - start_time) * 1000.0; // ms

    // 4. Process 0 aggregates and verifies
    if (rank == 0) {
        total_sum = 0;
        printf("-----------------------------------------------------------------\n");
        printf("[Root Process 0] Gathered Local Sums from Processes:\n");
        for (int p = 0; p < size; p++) {
            printf("   Process %2d Sum = %d\n", p, gathered_sums[p]);
            total_sum += gathered_sums[p];
        }

        long long expected_sum = ((long long)array_size * (array_size + 1)) / 2;
        printf("-----------------------------------------------------------------\n");
        printf("[Root Process 0] Final Global Sum: %lld (Expected: %lld)\n", total_sum, expected_sum);
        printf("[Verification] Status: %s\n", (total_sum == expected_sum) ? "PASSED (MATCHES MATHEMATICAL EXPECTATION)" : "FAILED");
        printf("[Performance] Total Scatter-Gather Cycle Time: %.6f ms\n", scatter_gather_time);
        printf("=================================================================\n\n");

        free(global_array);
        free(gathered_sums);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
