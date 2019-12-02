#include "Grid.h"

using namespace std;

int mainSerial(int argc, char **argv) {
    int N = atoi(argv[1]);;
    long int K = 1e11;
    double Lx = 1; 
    double Ly = 1;
    double T = 1;

    clock_t begin = clock();
    
    Grid gr(N, K, Lx, Ly, T);
    gr.init();

    for(int i = 0; i < 20; i++) {
        gr.makeStep();  
        double residual = gr.residual(); 
        cout << i << " : " << residual << endl;  
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "Time of serial execution on " << N << " is " << time_spent << endl;

    return 0;
}

int main(int argc, char **argv) {
    return mainSerial(argc, argv);
}