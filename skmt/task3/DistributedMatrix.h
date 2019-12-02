#include <cmath>
#include <omp.h>
#include "mpi.h"

using namespace std;

#define PI 3.14159265359

class DistributedMatrix
{
private:
    double **data0;
    double **data1;

    // Размеры локальной матрицы (столбцы, строки)
    int localX, localY;

    // Размеры глобальной матрицы
    int globalN, globalM;

    // MPI
    int myRank;
    int nProc;
    int neighbours[8];

    // Координаты процесса в сетке
    int myX, myY;

    // Кол-во процессов в строке/столбце
    int procX, procY;

    long int N, K;
    double Lx, Ly, T;
    double hx, hy, tau;
    int currT;

    int numThreads;

    double f0(double x, double y, double t);
    double f1(double x, double y, double t);
    double u0(double x, double y, double t);
    double u1(double x, double y, double t);
    void globalCoord(int i, int j, int &globI, int &globJ);

public:
    DistributedMatrix(int _N, int _K, double _Lx, double _Ly, double _T, int _nProc, int _myRank, int _procX, int _procY, int _numThreads);
    void initialize();
    void sync();
    double calcDelta();
    void makeIter();
    ~DistributedMatrix();
};

DistributedMatrix::DistributedMatrix(int _N, int _K, double _Lx, double _Ly, double _T, int _nProc, int _myRank, int _procX, int _procY, int _numThreads)
{
    N = _N;
    K = _K;
    Lx = _Lx;
    Ly = _Ly;
    T = _T;
    currT = 1;

    nProc = _nProc;
    myRank = _myRank;
    procX = _procX;
    procY = _procY;
    numThreads = _numThreads;

    myY = myRank / procX;
    myX = myRank % procY;

    tau = double(T) / K;
    hx = double(Lx) / N;
    hy = double(Ly) / N;

    localX = N / procX;
    localY = N / procY;
    if (myX == procX - 1)
        localX += N % procX;
    if (myY == procY - 1)
        localY += N % procY;
    
    data0 = new double* [localY + 2];
    data1 = new double* [localY + 2];
    for (int i = 0; i < localY + 2; i++) 
    {
        data0[i] = new double[localX + 2];
        data1[i] = new double[localX + 2];
    }
    
    // 0 1 2
    // 7 * 3
    // 6 5 4

    int x[8];
    int y[8];

    x[0] = myX - 1;
    y[0] = myY - 1;

    x[1] = myX;
    y[1] = myY - 1;
    
    x[2] = myX + 1;
    y[2] = myY - 1;
    
    x[3] = myX + 1;
    y[3] = myY;
    
    x[4] = myX + 1;
    y[4] = myY + 1;
    
    x[5] = myX;
    y[5] = myY + 1;
    
    x[6] = myX - 1;
    y[6] = myY + 1;
    
    x[7] = myX - 1;
    y[7] = myY;

    for(int i = 0; i < 8; i++)
    {
        if (x[i] < 0)
            x[i] = procX - 1;
        if (x[i] >= procX)
            x[i] = 0;
        if (y[i] < 0)
            y[i] = procY - 1;
        if (y[i] >= procY)
            y[i] = 0;

        neighbours[i] = x[i] + y[i] * procX;
    }

    if (myRank == 0)
        if (tau < min(hx * hx, hy * hy))
            cout << tau << " < " << min(hx * hx, hy * hy) << " => True" << endl;
        else
            cout << tau << " < " << min(hx * hx, hy * hy) << " => False" << endl;
}

DistributedMatrix::~DistributedMatrix()
{
    for (int i = 0; i < localY + 2; i++)
    {
        delete[] data0[i];
        delete[] data1[i];
    }
    delete[] data0;
    delete[] data1;
}

void DistributedMatrix::globalCoord(int i, int j, int &globI, int &globJ)
{
    globI = myY * N / procY + i;
    globJ = myX * N / procX + j;
}

double DistributedMatrix::f0(double x, double y, double t)
{
    // return sin(PI * x / Lx) * cos(t + 2 * PI) + 2 * (PI / Lx) * (PI / Lx) * sin(PI * x / Lx) * sin(t + 2 * PI);
    return sin(PI * x / Lx) * cos(t + 2 * PI) + 2 * (PI / Lx) * (PI / Lx) * sin(PI * x / Lx) * sin(t + 2 * PI) - 2 * PI * PI / (Lx * Ly) * cos(PI * x / Lx) * cos(2 * PI * y / Ly) * sin(t + 2 * PI);
}

double DistributedMatrix::f1(double x, double y, double t)
{
    // return sin(PI * x / Lx) * sin(2 * PI * y / Ly) * cos(t + 2 * PI) + PI * PI * (1 / (Lx * Lx) + 8 / (Ly * Ly)) * sin(PI * x / Lx) * sin(2 * PI * y / Ly) * sin(t + 2 * PI) - 2 * PI * PI / (Lx * Ly) * cos(PI * x / Lx) * cos(2 * PI * y / Ly) * sin(t + 2 * PI);
    return sin(PI * x / Lx) * sin(2 * PI * y / Ly) * cos(t + 2 * PI) + PI * PI * (1 / (Lx * Lx) + 8 / (Ly * Ly)) * sin(PI * x / Lx) * sin(2 * PI * y / Ly) * sin(t + 2 * PI);
}

double DistributedMatrix::u0(double x, double y, double t)
{
    return sin(PI * x / Lx) * sin(t + 2 * PI);
}

double DistributedMatrix::u1(double x, double y, double t)
{
    return sin(PI * x / Lx) * sin(2 * PI * y / Ly) * sin(t + 2 * PI);
}

void DistributedMatrix::initialize()
{
    int globI, globJ;

    for(int i = 1; i < localY + 1; i++)
        for(int j = 1; j < localX + 1; j++)
        {
            globalCoord(i - 1, j - 1, globI, globJ);
            data0[i][j] = u0(globJ * hx, globI * hy, 0);
            data1[i][j] = u1(globJ * hx, globI * hy, 0);
        }
}

void DistributedMatrix::sync()
{
    double tmpSend[2];
    double tmpRecv[2];

    double tmp2Send[2][localY];
    double tmp2Recv[2][localY];

    for(int i = 0; i < 8; i++)
    {
        if (i % 2 == 0)
        {
            int k = localY, l = 1;
            if (i == 0)
            {
                k = 1;
                l = 1;
            }
            else
            if (i == 2)
            {
                k = 1;
                l = localX;
            }
            else
            if (i == 4)
            {
                k = localY;
                l = localX;
            }

            if ((myY == 0) && ((i == 0) ||  (i == 2)))
                k++;

            tmpSend[0] = data0[k][l];
            tmpSend[1] = data1[k][l];
            
            MPI_Sendrecv(tmpSend, 2, MPI_DOUBLE, neighbours[i], 1, tmpRecv, 2, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            k = 0;
            l = localX + 1;
            if (i == 0)
            {
                k = localY + 1;
                l = localX + 1;
            }
            else
            if (i == 2)
            {
                k = localY + 1;
                l = 0;
            }
            else
            if (i == 4)
            {
                k = 0;
                l = 0;
            }

            data0[k][l] = tmpRecv[0];
            data1[k][l] = tmpRecv[1];
        } else
        if (i % 4 == 1)
        {
            if (i == 1)
            {
                int k = 1;
                if (myY == 0)
                    k = 2;

                MPI_Sendrecv(&data0[k][1], localX, MPI_DOUBLE, neighbours[i], 1, &data0[localY + 1][1], localX, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&data1[k][1], localX, MPI_DOUBLE, neighbours[i], 1, &data1[localY + 1][1], localX, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else // (i == 5)
            {
                MPI_Sendrecv(&data0[localY][1], localX, MPI_DOUBLE, neighbours[i], 1, &data0[0][1], localX, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(&data1[localY][1], localX, MPI_DOUBLE, neighbours[i], 1, &data1[0][1], localX, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else
        if (i % 4 == 3)
        {
            int k = localX;
            if (i == 7)
                k = 1;

            for(int j = 1; j < localY + 1; j++)
            {
                tmp2Send[0][j - 1] = data0[j][k];
                tmp2Send[1][j - 1] = data1[j][k];
            }

            MPI_Sendrecv(tmp2Send, localY * 2, MPI_DOUBLE, neighbours[i], 1, tmp2Recv, localY * 2, MPI_DOUBLE, neighbours[(4 + i) % 8], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            k = 0;
            if (i == 7)
                k = localX + 1;

            for(int j = 1; j < localY + 1; j++)
            {
                data0[j][k] = tmp2Recv[0][j - 1];
                data1[j][k] = tmp2Recv[1][j - 1];
            }
        }
    }
}

double DistributedMatrix::calcDelta()
{
    int globI, globJ;
    double localMax = 0.0;
    double globalMax = 0.0;

    for(int i = 1; i < localY + 1; i++)
        for(int j = 1; j < localX + 1; j++)
        {
            globalCoord(i - 1, j - 1, globI, globJ);
            localMax = max(abs(data0[i][j] - u0(globJ * hx, globI * hy, (currT - 1) * tau)), localMax);
            localMax = max(abs(data1[i][j] - u1(globJ * hx, globI * hy, (currT - 1) * tau)), localMax);
        }

    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(myRank == 0)
        cout << (currT - 1) << " : " << globalMax << endl;

    return globalMax;
}

void DistributedMatrix::makeIter()
{
    sync();

    // MPI_Barrier(MPI_COMM_WORLD);
    // for(int t = 0; t < nProc; t++)
    // {
    //     if(myRank == t)
    //     {
    //         cout << " < " << myRank << " > " << endl;
    //         // cout << "data0" << endl;
    //         // for(int i = localY + 1; i > -1; i--)
    //         // {
    //         //     for(int j = 0; j < localX + 2; j++)
    //         //         cout << data0[i][j] << " ";
    //         //     cout << endl;
    //         // }
    //         cout << "data1" << endl;
    //         // for(int i = localY + 1; i >= 0; i--)
    //         for(int i = 0; i < localY + 2; i++)
    //         {
    //             for(int j = 0; j < localX + 2; j++)
    //                 cout << data1[i][j] << " ";
    //             cout << endl;
    //         }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // // return;

    double** newData0 = new double* [localY + 2];
    double** newData1 = new double* [localY + 2];
    for(int i = 0; i < localY + 2; i++)
    {
        newData0[i] = new double[localX + 2];
        newData1[i] = new double[localX + 2];
    }

    int globI, globJ;
    for(int i = 1; i < localY + 1; i++)
        for(int j = 1; j < localX + 1; j++)
        {
            globalCoord(i - 1, j - 1, globI, globJ);
            if ((globJ == 0) || (globJ == N - 1))
            {
                // u1x - 1 рода, u2x - 1 рода
                newData0[i][j] = 0;
                newData1[i][j] = 0;
            } else
            if (globI == N - 1)
            {
                // u1y - 2 рода, u2y - периодические
                double laplasV0 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                laplasV0 += (data1[i - 1][j] - 2 * data1[i][j] + data1[i + 1][j]) / (hy * hy);
                double nablaDivV0 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                nablaDivV0 += (data0[i + 1][j + 1] - data0[i - 1][j + 1] - data0[i + 1][j - 1] + data0[i - 1][j - 1]) / (4 * hx * hy);

                newData1[i][j] = tau * (laplasV0 + nablaDivV0 + f1(globJ * hx, globI * hy, currT * tau)) + data1[i][j];

                newData0[i][j] = (4 * data0[i - 1][j] - data0[i - 2][j]) / 3; // 2 род
            } else
            if (globI == 0)
            {
                // u1y - 2 рода
                newData0[i][j] = (4 * data0[i + 1][j] - data0[i + 2][j]) / 3;
            } else
            {
                // Остальные точки
                double laplasV0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                laplasV0 += (data0[i - 1][j] - 2 * data0[i][j] + data0[i + 1][j]) / (hy * hy);
                double nablaDivV0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                nablaDivV0 += (data1[i + 1][j + 1] - data1[i - 1][j + 1] - data1[i + 1][j - 1] + data1[i - 1][j - 1]) / (4 * hx * hy);

                double laplasV1 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                laplasV1 += (data1[i - 1][j] - 2 * data1[i][j] + data1[i + 1][j]) / (hy * hy);
                double nablaDivV1 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                nablaDivV1 += (data0[i + 1][j + 1] - data0[i - 1][j + 1] - data0[i + 1][j - 1] + data0[i - 1][j - 1]) / (4 * hx * hy);

                newData0[i][j] = tau * (laplasV0 + nablaDivV0 + f0(globJ * hx, globI * hy, currT * tau)) + data0[i][j];
                newData1[i][j] = tau * (laplasV1 + nablaDivV1 + f1(globJ * hx, globI * hy, currT * tau)) + data1[i][j];
            }
        }

    for(int i = 0; i < localY + 2; i++)
    {
        delete[] data0[i];
        delete[] data1[i];
        data0[i] = newData0[i];
        data1[i] = newData1[i];
    }
    data0 = newData0;
    data1 = newData1;

    if (myY == procY - 1)
        MPI_Send(&data1[localY][1], localX, MPI_DOUBLE, neighbours[5], 0, MPI_COMM_WORLD);
    
    if (myY == 0)
        MPI_Recv(&data1[1][1], localX, MPI_DOUBLE, neighbours[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);

    currT++;
}