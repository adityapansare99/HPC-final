#include<bits/stdc++.h>
#include<mpi.h>
using namespace std;

int main(int argc,char *argv[]){
    int rank,size;
    MPI_Init(&argc,&argv);

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    cout<<"size is: "<<size<<"Rank is: "<<rank<<endl;

    MPI_Finalize();
}