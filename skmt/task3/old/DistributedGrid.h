#include <cmath>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <omp.h>
#include "mpi.h"

#define PI 3.14159265359

using namespace std;

// u1x - 2 рода, u1y - периодические, u2x - 1 рода, u2y - 2 рода, 
class DistributedGrid {
private:
    // Данные
    double **data0;
    double **data1;
    
    // Размеры локальной решетки
    int localX, localY;
    
    // MPI
    int myRank;
    int sizeGroup;
    int neighbours[8];
    
    // Координаты процесса в решетке процессов
    int procX, procY;
    
    // Кол-во процессов по каждой оси
    int numProcX, numProcY;

    int numThreads;

    // Данные решетки
    long int N, K;
    double Lx, Ly, T;
    double tau, hx, hy;
    int cntT;

    double F(double x, double y, double t, int ind);
    double u(double x, double y, double t, int ind);
    void getGlobalCoord(int i, int j, int &gI, int &gJ);

public:
    DistributedGrid(int _N, int _K, double _Lx, double _Ly, double _T, int _myRank, int _sizeGroup, int _numProcX, int _numProcY, int _numThreads);
    void init();
    void makeStep();
    double residual();
    ~DistributedGrid();
};

// _numProcX - кол-во процессов по оси X
// _numProcY - кол-во процессов по оси Y
DistributedGrid::DistributedGrid(int _N, int _K, double _Lx, double _Ly, double _T, int _myRank, int _sizeGroup, int _numProcX, int _numProcY, int _numThreads) {
    // Инициализация переменных
    N = _N;
    K = _K;
    Lx = _Lx;
    Ly = _Ly;
    T = _T;
    cntT = 1;
    numThreads = _numThreads;

    myRank = _myRank;
    sizeGroup = _sizeGroup;
    numProcX = _numProcX;
    numProcY = _numProcY;

    procY = myRank / numProcX;
    procX = myRank % numProcX;

    tau = double(T) / K;
    hx = double(Lx) / N;
    hy = double(Ly) / N;

    localX = N / numProcX;
    localY = N / numProcY;
    if (procX == numProcX - 1) localX += N % numProcX;
    if (procY == numProcY - 1) localY += N % numProcY;

    // Вычисление соседей
    double tmpX[8];
    double tmpY[8];

    for(int i = 0; i < 8; i++){
        tmpX[i] = -1;
        tmpY[i] = -1;
    }

    tmpX[0] = procX - 1;
    tmpY[0] = procY + 1;

    tmpX[1] = procX;
    tmpY[1] = procY + 1;
    
    tmpX[2] = procX + 1;
    tmpY[2] = procY + 1;
    
    tmpX[3] = procX + 1;
    tmpY[3] = procY;
    
    tmpX[4] = procX + 1;
    tmpY[4] = procY - 1;
    
    tmpX[5] = procX;
    tmpY[5] = procY - 1;
    
    tmpX[6] = procX - 1;
    tmpY[6] = procY - 1;
    
    tmpX[7] = procX - 1;
    tmpY[7] = procY;

    if ((procX == 0) && (procY != 0) && (procY != numProcY - 1)) {
        tmpX[0] = tmpY[0] = -1;
        tmpX[7] = tmpY[7] = -1;
        tmpX[6] = tmpY[6] = -1;
    }

    if ((procX == numProcX - 1) && (procY != 0) && (procY != _numProcY - 1)) {
        tmpX[2] = tmpY[2] = -1;
        tmpX[3] = tmpY[3] = -1;
        tmpX[4] = tmpY[4] = -1;
    }

    if ((procY == 0) && (procX != 0) && (procX != numProcX - 1)) {
        tmpX[4] = procX + 1;
        tmpY[4] = numProcY - 1;
        tmpX[6] = procX - 1;
        tmpY[6] = numProcY - 1;
        tmpX[5] = procX;
        tmpY[5] = numProcY - 1; 
    }

    if ((procY == numProcY - 1) && (procX != 0) && (procX != numProcX - 1)) {
        tmpX[0] = procX - 1;
        tmpY[0] = 0;
        tmpX[2] = procX + 1;
        tmpY[2] = 0;
        tmpX[1] = procX;
        tmpY[1] = 0;
    }

    if ((procX == 0) && (procY == 0)) {
        tmpX[4] = procX + 1;
        tmpY[4] = numProcY - 1;
        tmpX[0] = tmpY[0] = -1;
        tmpX[7] = tmpY[7] = -1;
        tmpX[6] = tmpY[6] = -1;
        tmpX[5] = procX;
        tmpY[5] = numProcY - 1; 
    }

    if ((procX == numProcX - 1) && (procY == 0)) {
        tmpX[6] = procX - 1;
        tmpY[6] = numProcY - 1;
        tmpX[2] = tmpY[2] = -1;
        tmpX[3] = tmpY[3] = -1;
        tmpX[4] = tmpY[4] = -1;
        tmpX[5] = procX;
        tmpY[5] = numProcY - 1; 
    }

    if ((procX == 0) && (procY == numProcY - 1)) {
        tmpX[0] = tmpY[0] = -1;
        tmpX[6] = tmpY[6] = -1;
        tmpX[7] = tmpY[7] = -1;
        tmpX[1] = procX;
        tmpY[1] = 0;
        tmpX[2] = procX + 1;
        tmpY[2] = 0;
    }

    if ((procX == numProcX - 1) && (procY == numProcY - 1)) {
        tmpX[0] = procX - 1;
        tmpY[0] = 0;
        tmpX[2] = tmpY[2] = -1;
        tmpX[3] = tmpY[3] = -1;
        tmpX[4] = tmpY[4] = -1;
        tmpX[1] = procX;
        tmpY[1] = 0;
    }   

    for(int i = 0; i < 8; i++) {
        if (tmpX[i] != -1) 
            neighbours[i] = tmpY[i] * numProcX + tmpX[i];
        else
            neighbours[i] = -1;
    }
    
    // Выделение памяти под данные
    data0 = new double*[localY + 2];
    data1 = new double*[localY + 2];
    for(int i = 0; i < localY + 2; i++) {
        data0[i] = new double[localX + 2];
        data1[i] = new double[localX + 2];
        double nans = nan("");
        for(int j = 0; j < localX + 2; j++) {
            data0[i][j] = nans;
            data1[i][j] = nans;
        }
    }

    if (myRank == 0)
        cout << tau << ", " << min(hx * hx, hy * hy) << endl;
    // cout << myRank << " : " << neighbours[0] << " " << neighbours[1] << " " << neighbours[2] << " " << neighbours[3] << " " << neighbours[4] << " " << neighbours[5] << " " << neighbours[6] << " " << neighbours[7] << endl;
}

double DistributedGrid::residual() {
    int maxI, maxJ;
    double m = 0.0;
    for(int i = 1; i < localY + 1; i++)
        for(int j = 1; j < localX + 1; j++) {
            int gI, gJ;
            getGlobalCoord(i - 1, j - 1, gI, gJ);
            m = max(abs(data0[i][j] - u(gJ * hx, gI * hy, cntT * tau, 0)), m);
            m = max(abs(data1[i][j] - u(gJ * hx, gI * hy, cntT * tau, 1)), m);
        }
    double global_max;
    MPI_Reduce(&m, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return global_max;
}

// Работает только с точками за которые ответчает даный процесс (гало не работает)
void DistributedGrid::getGlobalCoord(int i, int j, int &gI, int &gJ) {
    gI = procY * (N / numProcY) + i;
    gJ = procX * (N / numProcX) + j;
}

double DistributedGrid::F(double x, double y, double t, int ind) {
    if (ind == 0)
        return -sin(t + 2 * PI) * cos(PI * x / Lx) - 2 * (PI / Lx) * (PI / Lx) * cos(PI * x / Lx) * cos(t + 2 * PI);
    else
        return -sin(t + 2 * PI) * sin(PI * x / Lx) * cos(PI * y / Ly) - (-(PI/Lx) * (PI/Lx)  - 2 * (PI/Ly) * (PI/Ly)) * sin(PI * x / Lx) * cos(PI * y / Ly) * cos(t + 2 * PI) - PI * PI / (Lx * Ly) * sin(PI * x / Lx) * sin(PI * y / Ly) * cos(t + 2 * PI);
}

double DistributedGrid::u(double x, double y, double t, int ind) {
    if (ind == 0)
        return cos(PI * x / Lx) * cos(t + 2 * PI);
    else
        return sin(PI * x / Lx) * cos(PI * y / Ly) * cos(t + 2 * PI);
}

void DistributedGrid::init() {
    for(int i = 1; i < localY + 1; i++) {
        for(int j = 1; j < localX + 1; j++) {
            int gI, gJ;
            getGlobalCoord(i - 1, j - 1, gI, gJ);
            data0[i][j] = u(gJ * hx, gI * hy, 0, 0);
            data1[i][j] = u(gJ * hx, gI * hy, 0, 1);
        }
    }
}

void DistributedGrid::makeStep() {
    // MPI_Barrier(MPI_COMM_WORLD);
    // for(int t = 0; t < sizeGroup; t++) {
    //     if(myRank == t) {
    //         cout << " << " << myRank << " >> " << endl;
    //         cout << "data0" << endl;
    //         for(int i = localY + 1; i > -1; i--) {
    //             for(int j = 0; j < localX + 2; j++)
    //                 cout << data0[i][j] << " ";
    //             cout << endl;
    //         }
    //         // cout << "data1" << endl;
    //         // for(int i = localY + 1; i >= 0; i--) {
    //         //     for(int j = 0; j < localX + 2; j++)
    //         //         cout << data1[i][j] << " ";
    //         //     cout << endl;
    //         // }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    // Выделение и заполнение буферов
    double* bufferSnd[8];
    double* bufferRcv[8];
    
    if(neighbours[0] != -1) {
        bufferSnd[0] = new double[2];
        bufferRcv[0] = new double[2];
        bufferSnd[0][0] = data0[localY][1];
        bufferSnd[0][1] = data1[localY][1];
    }

    if(neighbours[1] != -1) {
        bufferSnd[1] = new double[2 * localX];
        bufferRcv[1] = new double[2 * localX];
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[1][i - 1] = data0[localY][i];
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[1][localX - 1 + i] = data1[localY][i];
    }

    if(neighbours[2] != -1) {
        bufferSnd[2] = new double[2];
        bufferRcv[2] = new double[2];
        bufferSnd[2][0] = data0[localY][localX];
        bufferSnd[2][1] = data1[localY][localX];
    }

    if(neighbours[3] != -1) {
        bufferSnd[3] = new double[2 * localY];
        bufferRcv[3] = new double[2 * localY];
        for (int i = 1; i < localY + 1; i++)
            bufferSnd[3][i - 1] = data0[i][localX];
        for (int i = 1; i < localY + 1; i++)
            bufferSnd[3][localY - 1 + i] = data1[i][localX];
    }
    
    if(neighbours[4] != -1) {
        bufferSnd[4] = new double[2];
        bufferRcv[4] = new double[2];
        bufferSnd[4][0] = data0[1][localX];
        bufferSnd[4][1] = data1[1][localX];
    }

    if(neighbours[5] != -1) {
        bufferSnd[5] = new double[2 * localX];
        bufferRcv[5] = new double[2 * localX];
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[5][i - 1] = data0[1][i];
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[5][localX - 1 + i] = data1[1][i];
    }

    if(neighbours[6] != -1) {
        bufferSnd[6] = new double[2];
        bufferRcv[6] = new double[2];
        bufferSnd[6][0] = data0[1][1];
        bufferSnd[6][1] = data1[1][1];
    }

    if(neighbours[7] != -1) {
        bufferSnd[7] = new double[2 * localY];
        bufferRcv[7] = new double[2 * localY];
        for (int i = 1; i < localY + 1; i++)
            bufferSnd[7][i - 1] = data0[i][1];
        for (int i = 1; i < localY + 1; i++)
            bufferSnd[7][localY - 1 + i] = data1[i][1];
    }

    // Обработка буферов для периодического условия (TODO перенести в верхний if)
    if (procY == 0) {
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[5][i - 1] = data0[2][i];
        for (int i = 1; i < localX + 1; i++)
            bufferSnd[5][localX - 1 + i] = data1[2][i];
        
        if (procX == 0) {
            bufferSnd[4][0] = data0[2][localX];
            bufferSnd[4][1] = data1[2][localX];
        } else if (procX == numProcX - 1) {
            bufferSnd[6][0] = data0[2][1];
            bufferSnd[6][1] = data1[2][1];
        } else {
            bufferSnd[4][0] = data0[2][localX];
            bufferSnd[4][1] = data1[2][localX];
            bufferSnd[6][0] = data0[2][1];
            bufferSnd[6][1] = data1[2][1];
        }
    }

    // Обмен данными
    MPI_Request requestsSnd[8];
    MPI_Request requestsRcv[8];

    for(int i = 0; i < 8; i++) {
        if(neighbours[i] == -1)
            continue;
        
        if ((procY != numProcY - 1) || ((i != 1) && (i != 0) && (i != 2))) {
            if ((i == 0) || (i == 2) || (i == 4) || (i == 6))
                MPI_Isend(bufferSnd[i], 2, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsSnd[i]);
            else if((i == 7) || (i == 3))
                MPI_Isend(bufferSnd[i], 2 * localY, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsSnd[i]);
            else
                MPI_Isend(bufferSnd[i], 2 * localX, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsSnd[i]);
        }

        if ((procY != 0) || ((i != 5) && (i != 4) && (i != 6))) {
            if ((i == 0) || (i == 2) || (i == 4) || (i == 6))
                MPI_Irecv(bufferRcv[i], 2, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsRcv[i]);
            else if((i == 7) || (i == 3))
                MPI_Irecv(bufferRcv[i], 2 * localY, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsRcv[i]);
            else
                MPI_Irecv(bufferRcv[i], 2 * localX, MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, &requestsRcv[i]);
        }
    }
    
    for(int i = 0; i < 8; i++) {
        if(neighbours[i] == -1)
            continue;
        if ((procY == numProcY - 1) && ((i == 1) || (i == 0) || (i == 2)))
            continue;
        
        MPI_Wait(&requestsSnd[i], MPI_STATUS_IGNORE);
    }

    for(int i = 0; i < 8; i++) {
        if(neighbours[i] == -1)
            continue;
        if ((procY == 0) && ((i == 5) || (i == 4) || (i == 6)))
            continue;
        
        MPI_Wait(&requestsRcv[i], MPI_STATUS_IGNORE);
    }

    if(neighbours[0] != -1) {
        data0[localY + 1][0] = bufferRcv[0][0];
        data1[localY + 1][0] = bufferRcv[0][1];
    }

    if(neighbours[1] != -1) {
        for (int i = 1; i < localX + 1; i++)
            data0[localY + 1][i] = bufferRcv[1][i - 1];
        for (int i = 1; i < localX + 1; i++)
            data1[localY + 1][i] = bufferRcv[1][localX - 1 + i];
    }

    if(neighbours[2] != -1) {
        data0[localY + 1][localX + 1] = bufferRcv[2][0];
        data1[localY + 1][localX + 1] = bufferRcv[2][1];
    }

    if(neighbours[3] != -1) {
        for (int i = 1; i < localY + 1; i++)
            data0[i][localX + 1] = bufferRcv[3][i - 1];
        for (int i = 1; i < localY + 1; i++)
            data1[i][localX + 1] = bufferRcv[3][localY - 1 + i];
    }
    
    if(neighbours[4] != -1) {
        data0[0][localX + 1] = bufferRcv[4][0];
        data1[0][localX + 1] = bufferRcv[4][1];
    }

    if(neighbours[5] != -1) {
        for (int i = 1; i < localX + 1; i++)
            data0[0][i] = bufferRcv[5][i - 1];
        for (int i = 1; i < localX + 1; i++)
            data1[0][i] = bufferRcv[5][localX - 1 + i];
    }

    if(neighbours[6] != -1) {
        data0[0][0] = bufferRcv[6][0];
        data1[0][0] = bufferRcv[6][1];
    }

    if(neighbours[7] != -1) {
        for (int i = 1; i < localY + 1; i++)
            data0[i][0] = bufferRcv[7][i - 1];
        for (int i = 1; i < localY + 1; i++)
            data1[i][0] = bufferRcv[7][localY - 1 + i];
    }

    // MPI_Barrier(MPI_COMM_WORLD);
    // for(int t = 0; t < sizeGroup; t++) {
    //     if(myRank == t) {
    //         cout << " << " << myRank << " >> " << endl;
    //         cout << "data0" << endl;
    //         for(int i = localY + 1; i >= 0; i--) {
    //             for(int j = 0; j < localX + 2; j++)
    //                 cout << data0[i][j] << " ";
    //             cout << endl; 
    //         }
    //         // cout << "data1" << endl;
    //         // for(int i = localY + 1; i >= 0; i--) {
    //         //     for(int j = 0; j < localX + 2; j++)
    //         //         cout << data1[i][j] << " ";
    //         //     cout << endl;
    //         // }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
    // exit(0);

    // Вычисление значений в точках решетки
    double** newData0 = new double*[localY + 2];
    double** newData1 = new double*[localY + 2];
    for(int i = 0; i < localY + 2; i++) {
        newData0[i] = new double[localX + 2];
        newData1[i] = new double[localX + 2];
    }

    omp_set_num_threads(numThreads);
    
    #pragma omp parallel for
    for(int i = 1; i < localY + 1; i++)
        for(int j = 1; j < localX + 1; j++) {
            int gI, gJ;
            getGlobalCoord(i - 1, j - 1, gI, gJ);
            if (gJ == 0) {
                // u1x - 2 рода, u2x - 1 рода
                newData0[i][j] = (4 * data0[i][j + 1] - data0[i][j + 2]) / 3;
                newData1[i][j] = 0;
            } else if (gJ == N - 1) {
                // u1x - 2 рода, u2x - 1 рода
                newData0[i][j] = (4 * data0[i][j - 1] - data0[i][j - 2]) / 3;
                newData1[i][j] = 0;
            } else if (gI == N - 1) {
                // u1y - периодические, u2y - 2 рода
                double delta0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                delta0 += (data0[i - 1][j] - 2 * data0[i][j] + data0[i + 1][j]) / (hy * hy);
                double nablaDivV0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                nablaDivV0 += (data1[i + 1][j + 1] - data1[i - 1][j + 1] - data1[i + 1][j - 1] + data1[i - 1][j - 1]) / (4 * hx * hy) ;
                newData0[i][j] = tau * (delta0 + nablaDivV0 + F(gJ * hx, gI * hy, cntT * tau, 0)) + data0[i][j];

                newData1[i][j] = (4 * data1[i - 1][j] - data1[i - 2][j]) / 3;
            } else if (gI == 0) {
                // u2y - 2 рода
                newData1[i][j] = (4 * data1[i + 1][j] - data1[i + 2][j]) / 3;
            } else {
                // Остальные точки
                double delta0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                delta0 += (data0[i - 1][j] - 2 * data0[i][j] + data0[i + 1][j]) / (hy * hy);
                double nablaDivV0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                nablaDivV0 += (data1[i + 1][j + 1] - data1[i - 1][j + 1] - data1[i + 1][j - 1] + data1[i - 1][j - 1]) / (4 * hx * hy) ;

                double delta1 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                delta1 += (data1[i - 1][j] - 2 * data1[i][j] + data1[i + 1][j]) / (hy * hy);
                double nablaDivV1 = (data1[i][j - 1] - 2 * data1[i][j] + data1[i][j + 1]) / (hx * hx);
                nablaDivV1 += (data0[i + 1][j + 1] - data0[i - 1][j + 1] - data0[i + 1][j - 1] + data0[i - 1][j - 1]) / (4 * hx * hy);

                newData0[i][j] = tau * (delta0 + nablaDivV0 + F(gJ * hx, gI * hy, cntT * tau, 0)) + data0[i][j];
                newData1[i][j] = tau * (delta1 + nablaDivV1 + F(gJ * hx, gI * hy, cntT * tau, 1)) + data1[i][j];
            }
        }

    for(int i = 0; i < localY + 2; i++) {
        delete[] data0[i];
        delete[] data1[i];
        data0[i] = newData0[i];
        data1[i] = newData1[i];
    }
    data0 = newData0;
    data1 = newData1;

    // Пересылка периодических границ нижнему ряду
    if (procY == numProcY - 1)
        MPI_Send(&(data0[localY][1]), localX, MPI_DOUBLE, neighbours[1], 0, MPI_COMM_WORLD);
    
    if(procY == 0)
        MPI_Recv(&(data0[1][1]), localX, MPI_DOUBLE, neighbours[5], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);

    cntT += 1;
}

DistributedGrid::~DistributedGrid() {
    for(int i = 0; i < localY + 2; i++) {
        delete[] data0[i];
        delete[] data1[i];
    }
    delete[] data0;
    delete[] data1;
}

