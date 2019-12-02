#include "mpi.h"
#include "DistributedGrid.h"

using namespace std;

int mainMPI(int argc, char **argv) {
    int myRank, sizeGroup;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
    MPI_Comm_size(MPI_COMM_WORLD,&sizeGroup);

    int N = atoi(argv[1]);
    long int K = 1e11;
    double Lx = 1; 
    double Ly = 1;
    double T = 1;
    int numProcX = atoi(argv[2]);
    int numProcY = atoi(argv[3]);
    int numThreads = atoi(argv[4]);

    double time = MPI_Wtime();

    DistributedGrid dm(N, K, Lx, Ly, T, myRank, sizeGroup, numProcX, numProcY, numThreads);
    dm.init();
    
    for(int i = 0; i < 20; i++) { 
        double residual = dm.residual(); 
        if(myRank == 0)
            cout << i << " : " << residual << endl;
        dm.makeStep();   
    }

    time = MPI_Wtime() - time;
    double tmp;
    MPI_Reduce(&time, &tmp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myRank == 0)
        cout << "Time of parallel execution on " << N << " with " << sizeGroup << " procceses (" << numThreads << " each) is " << tmp << endl;

    MPI_Finalize();
    return 0;
}

int main(int argc, char **argv) {
    return mainMPI(argc, argv);
}