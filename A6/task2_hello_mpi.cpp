#include<bits/stdc++.h>
#include <mpi.h>
using namespace std;

/**
 * Assignment 6 - Task 2: First MPI Program
 * Description: Basic MPI initialization and termination demonstration.
 *              Displays "Hello from MPI" across all participating processes.
 */

int main(int argc, char *argv[]) {
    // Initialize the MPI execution environment
    // Allocates internal data structures and establishes communication channels.
    MPI_Init(&argc, &argv);

    // Each MPI process executes this independent statement concurrently
    cout<<"hello world"<<endl;

    // Clean up all MPI internal state, flush buffers, and terminate communication.
    MPI_Finalize();
    return 0;
}
