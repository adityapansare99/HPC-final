#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <windows.h>

void execute_task_A(int *data_A) {
    printf("[Thread %d] Task A: Initializing data...\n", omp_get_thread_num());
    *data_A = 100;
    Sleep(50); // Simulate work
}

void execute_task_B(int input_A, int *data_B) {
    printf("[Thread %d] Task B: Computing transform 1 from Task A (%d)...\n", omp_get_thread_num(), input_A);
    *data_B = input_A * 2;
    Sleep(50);
}

void execute_task_C(int input_A, int *data_C) {
    printf("[Thread %d] Task C: Computing transform 2 from Task A (%d)...\n", omp_get_thread_num(), input_A);
    *data_C = input_A + 50;
    Sleep(50);
}

void execute_task_D(int input_B, int input_C, int *final_result) {
    printf("[Thread %d] Task D: Aggregating Task B (%d) and Task C (%d)...\n", omp_get_thread_num(), input_B, input_C);
    *final_result = input_B + input_C;
    Sleep(50);
}

int main() {
    printf("========================================================================\n");
    printf("   Assignment 5 Advanced Exploration: OpenMP Task Dependencies (DAG)   \n");
    printf("========================================================================\n");
    printf("Computation Graph: Task A -> (Task B & Task C) -> Task D\n\n");

    int var_A = 0, var_B = 0, var_C = 0, final_result = 0;
    omp_set_num_threads(4);

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task A produces var_A
            #pragma omp task depend(out: var_A)
            execute_task_A(&var_A);

            // Task B consumes var_A, produces var_B
            #pragma omp task depend(in: var_A) depend(out: var_B)
            execute_task_B(var_A, &var_B);

            // Task C consumes var_A, produces var_C (runs concurrently with Task B!)
            #pragma omp task depend(in: var_A) depend(out: var_C)
            execute_task_C(var_A, &var_C);

            // Task D consumes var_B and var_C, produces final_result (waits for B and C!)
            #pragma omp task depend(in: var_B, var_C) depend(out: final_result)
            execute_task_D(var_B, var_C, &final_result);
        }
    }
    double elapsed = omp_get_wtime() - t0;

    printf("\nExecution completed successfully in %.6f seconds.\n", elapsed);
    printf("Final Aggregated Result (Task D): %d (Expected: 250)\n", final_result);
    return 0;
}
