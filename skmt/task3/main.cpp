#include "mpi.h"
#include "DistributedMatrix.h"
#include <stdlib.h>

using namespace std;

// mpicxx main.cpp -o main && mpirun -np 9 main 2048 3 3 2
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nProc = 0, myRank = 0;
    // MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // omp_set_num_threads(atoi(argv[6]));

    int N = atoi(argv[1]);
    long int K = 1e11;
    // long int K = 1000000000;
    double Lx = 1; 
    double Ly = 1;
    double T = 1;
    int procX = atoi(argv[2]);
    int procY = atoi(argv[3]);
    int numThreads = atoi(argv[4]);

    DistributedMatrix grid(N, K, Lx, Ly, T, nProc, myRank, procX, procY, numThreads);
    grid.initialize();
    grid.calcDelta();

    for (int i = 0; i < 10; i++)
    {
        grid.makeIter();
        grid.calcDelta();
    }

    MPI_Finalize();
    return 0;
}
