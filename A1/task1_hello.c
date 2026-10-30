#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    printf("=========================================================\n");
    printf("   Task 1: OpenMP Environment & Thread Verification      \n");
    printf("=========================================================\n");
    printf("Configured requested threads: %d\n\n", num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        printf("Hello World from Thread %d of %d!\n", tid, total_threads);
    }

    printf("\nTask 1 execution completed successfully.\n");
    return 0;
}

// gcc -fopenmp -O2 task1_hello.c -o task1_hello
// ./task1_hello 4