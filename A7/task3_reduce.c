/*
 * ==============================================================================
 * HPC Experiment 7 - Task 3: Global Reduction Using MPI_Reduce()
 * ==============================================================================

 * Description:
 *   Demonstrates MPI collective reductions:
 *   - MPI_Reduce(..., MPI_SUM, ...) to compute the global sum.
 *   - MPI_Reduce(..., MPI_MAX, ...) to find the global maximum.
 *   - MPI_Reduce(..., MPI_MIN, ...) to find the global minimum.
 *   - Compares performance & memory efficiency against MPI_Gather().
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>

#define DEFAULT_ARRAY_SIZE 16

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size = DEFAULT_ARRAY_SIZE;
    int *global_array = NULL;
    int *local_array = NULL;
    int chunk_size = 0;

    int local_sum = 0, local_min = INT_MAX, local_max = INT_MIN;
    int global_sum = 0, global_min = 0, global_max = 0;
    double start_time, end_time, reduce_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        array_size = atoi(argv[1]);
    }

    if (array_size % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Array size (%d) must be evenly divisible by process count (%d).\n",
                    array_size, size);
        }
        MPI_Finalize();
        return 1;
    }

    chunk_size = array_size / size;
    local_array = (int *)malloc(chunk_size * sizeof(int));

    if (rank == 0) {
        global_array = (int *)malloc(array_size * sizeof(int));
        printf("=================================================================\n");
        printf("       MPI Task 3: Global Reduction Using MPI_Reduce()           \n");
        printf("=================================================================\n");
        printf("[Root Process 0] Total Elements = %d | Processes = %d | Chunk/Process = %d\n",
               array_size, size, chunk_size);
        printf("[Root Process 0] Global Array: [");
        for (int i = 0; i < array_size; i++) {
            global_array[i] = i + 1; // 1, 2, ..., 16
            if (i < 16) {
                printf("%d%s", global_array[i], (i == array_size - 1 || i == 15) ? "" : ", ");
            }
        }
        if (array_size > 16) printf(", ... (%d items)", array_size);
        printf("]\n");
        printf("-----------------------------------------------------------------\n");
    }

    // Scatter the array
    MPI_Scatter(global_array, chunk_size, MPI_INT,
                local_array, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Compute local statistics
    local_sum = 0;
    local_min = local_array[0];
    local_max = local_array[0];

    for (int i = 0; i < chunk_size; i++) {
        int val = local_array[i];
        local_sum += val;
        if (val < local_min) local_min = val;
        if (val > local_max) local_max = val;
    }

    printf("[Process Rank %2d] Local Chunk [%d..%d] -> Local Sum=%d, Local Min=%d, Local Max=%d\n",
           rank, local_array[0], local_array[chunk_size - 1], local_sum, local_min, local_max);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Perform Collective Reductions
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    reduce_time = (end_time - start_time) * 1000.0; // ms

    if (rank == 0) {
        long long expected_sum = ((long long)array_size * (array_size + 1)) / 2;
        printf("-----------------------------------------------------------------\n");
        printf("[Root Process 0] Reduction Results Aggregated from %d Processes:\n", size);
        printf("   - Global Sum : %d (Expected: %lld) -> %s\n",
               global_sum, expected_sum, (global_sum == expected_sum) ? "PASSED" : "FAILED");
        printf("   - Global Min : %d (Expected: 1)     -> %s\n",
               global_min, (global_min == 1) ? "PASSED" : "FAILED");
        printf("   - Global Max : %d (Expected: %d)    -> %s\n",
               global_max, array_size, (global_max == array_size) ? "PASSED" : "FAILED");
        printf("-----------------------------------------------------------------\n");
        printf("[Performance] Total MPI_Reduce Time (Sum+Min+Max): %.6f ms\n", reduce_time);
        printf("[Architectural Advantage] Communication complexity is O(log P) vs O(P) for Gather.\n");
        printf("                          Root buffer requirement is O(1) vs O(P) for Gather.\n");
        printf("=================================================================\n\n");

        free(global_array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
