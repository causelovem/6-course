#include <cmath>
#include <algorithm>
#include <iostream>

#define PI 3.14159265359

using namespace std;

// u1x - 2 рода, u1y - периодические, u2x - 1 рода, u2y - 2 рода, 
class Grid {
private:
    // Данные
    double **data0;
    double **data1;

    // Данные решетки
    long int N, K;
    double Lx, Ly, T;
    double tau, hx, hy;
    int cntT;

    double F(double x, double y, double t, int ind);
    double u(double x, double y, double t, int ind);

public:
    Grid(int _N, int _K, double _Lx, double _Ly, double _T);
    void init();
    void makeStep();
    double residual();
    ~Grid();
};

Grid::Grid(int _N, int _K, double _Lx, double _Ly, double _T) {
    // Инициализация переменных
    N = _N;
    K = _K;
    Lx = _Lx;
    Ly = _Ly;
    T = _T;
    cntT = 1;

    tau = double(T) / K;
    hx = double(Lx) / N;
    hy = double(Ly) / N;
    
    // Выделение памяти под данные
    data0 = new double*[N];
    data1 = new double*[N];
    for(int i = 0; i < N; i++) {
        data0[i] = new double[N];
        data1[i] = new double[N];
    }

    cout << tau << ", " << min(hx * hx, hy * hy) << endl;
}

double Grid::residual() {
    double m = 0.0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) {
            m = max(abs(data0[i][j] - u(j * hx, i * hy, cntT * tau, 0)), m);
            m = max(abs(data1[i][j] - u(j * hx, i * hy, cntT * tau, 1)), m);
        }
    return m;
}

double Grid::F(double x, double y, double t, int ind) {
    if (ind == 0)
        return -sin(t + 2 * PI) * cos(PI * x / Lx) - 2 * (PI / Lx) * (PI / Lx) * cos(PI * x / Lx) * cos(t + 2 * PI);
    else
        return -sin(t + 2 * PI) * sin(PI * x / Lx) * cos(PI * y / Ly) - (-(PI/Lx) * (PI/Lx)  - 2 * (PI/Ly) * (PI/Ly)) * sin(PI * x / Lx) * cos(PI * y / Ly) * cos(t + 2 * PI) - PI * PI / (Lx * Ly) * sin(PI * x / Lx) * sin(PI * y / Ly) * cos(t + 2 * PI);
}

double Grid::u(double x, double y, double t, int ind) {
    if (ind == 0)
        return cos(PI * x / Lx) * cos(t + 2 * PI);
    else
        return sin(PI * x / Lx) * cos(PI * y / Ly) * cos(t + 2 * PI);
}

void Grid::init() {
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) {
            data0[i][j] = u(j * hx, i * hy, 0, 0);
            data1[i][j] = u(j * hx, i * hy, 0, 1);
        }
}

void Grid::makeStep() {
    // Вычисление значений в точках решетки
    double** newData0 = new double*[N];
    double** newData1 = new double*[N];
    for(int i = 0; i < N; i++) {
        newData0[i] = new double[N];
        newData1[i] = new double[N];
    }

    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) {
            if (j == 0) {
                // u1x - 2 рода, u2x - 1 рода
                newData0[i][j] = (4 * data0[i][j + 1] - data0[i][j + 2]) / 3;
                newData1[i][j] = 0;
            } else if (j == N - 1) {
                // u1x - 2 рода, u2x - 1 рода
                newData0[i][j] = (4 * data0[i][j - 1] - data0[i][j - 2]) / 3;
                newData1[i][j] = 0;
            } else if (i == N - 1) {
                // u1y - периодические, u2y - 2 рода
                double delta0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                delta0 += (data0[i - 1][j] - 2 * data0[i][j] + data0[1][j]) / (hy * hy);
                double nablaDivV0 = (data0[i][j - 1] - 2 * data0[i][j] + data0[i][j + 1]) / (hx * hx);
                nablaDivV0 += (data1[1][j + 1] - data1[i - 1][j + 1] - data1[1][j - 1] + data1[i - 1][j - 1]) / (4 * hx * hy) ;
                newData0[i][j] = tau * (delta0 + nablaDivV0 + F(j * hx, i * hy, cntT * tau, 0)) + data0[i][j];

                newData1[i][j] = (4 * data1[i - 1][j] - data1[i - 2][j]) / 3;
            } else if (i == 0) {
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

                newData0[i][j] = tau * (delta0 + nablaDivV0 + F(j * hx, i * hy, cntT * tau, 0)) + data0[i][j];
                newData1[i][j] = tau * (delta1 + nablaDivV1 + F(j * hx, i * hy, cntT * tau, 1)) + data1[i][j];
            }
        }

    for(int i = 0; i < N; i++) {
        delete[] data0[i];
        delete[] data1[i];
        data0[i] = newData0[i];
        data1[i] = newData1[i];
    }
    data0 = newData0;
    data1 = newData1;

    for(int i = 1; i < N-1; i++) {
        data0[0][i] = data0[N - 1][i];
    }

    cntT += 1;
}

Grid::~Grid() {
    for(int i = 0; i < N; i++) {
        delete[] data0[i];
        delete[] data1[i];
    }
    delete[] data0;
    delete[] data1;
}

