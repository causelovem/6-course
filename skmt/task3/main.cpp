#include "mpi.h"
#include "DistributedMatrix.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

// mpicxx -fopenmp main.cpp -o main && mpirun -np 9 main 2048 3 3 2
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nProc = 0, myRank = 0;
    // MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // long int K = 1e11;
    double K = 1e11;
    double Lx = 1; 
    double Ly = 1;
    double T = 1;

    int N = atoi(argv[1]);
    int procX = atoi(argv[2]);
    int procY = atoi(argv[3]);
    int numThreads = atoi(argv[4]);

    omp_set_num_threads(numThreads);

    DistributedMatrix grid(N, K, Lx, Ly, T, nProc, myRank, procX, procY);
    grid.initialize();
    grid.calcDelta();

    double time = MPI_Wtime();
    for (int i = 0; i < 20; i++)
    {
        grid.makeIter();
        grid.calcDelta();
    }
    time = MPI_Wtime() - time;

    double maxTime;
    MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRank == 0)
    {
        cout << "> Dim = " << N << endl;
        cout << "> Number of proc = " << nProc << endl;
        cout << "> Number of threads = " << numThreads << endl;
        cout << "> Max time = " << maxTime << endl;
    }

    MPI_Finalize();
    return 0;
}
