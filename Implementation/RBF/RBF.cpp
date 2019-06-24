//
// Created by thuchard on 12/06/2019.
//

#if _WIN32
#define SUPEREXPORT __declspec(dllexport)
#else
#define SUPEREXPORT
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <ctime>

#include "../Librairie/Matrix.h"

extern "C" {

int main() {
    double X[10] = {0.19503705,
                      0.5114616 ,
                      0.82267886,
                      0.5032035 ,
                      0.98414799,
                      0.15712639,
                      0.51985008,
                      0.51160201,
                      0.29984946,
                      0.71015867};
    double Y[] = { 0.53368513 };


//    0.88006435,
//    0.65887012,
//    0.95164382,
//    0.66102254,
//    0.58562491,
//    0.45204745,
//    0.38493358,
//    0.22059383,
//    0.1615526

    int epochs = 100;
    double gamma = 1;
    int rowsOfX = 10;
    int colsOfX = 1;
    int rowsOfY = 1;
    int colsOfY = 1;

    double phi[rowsOfX][colsOfX];

    std::cout << "Nombre de colonnes pour X : " << rowsOfX << std::endl;
    std::cout << "Nombre de lignes pour X : " << colsOfX << std::endl;
    std::cout << "Nombre de colonnes pour Y : " << rowsOfY << std::endl;
    std::cout << "Nombre de lignes pour Y : " << colsOfY << std::endl;

    std::cout << "Affichage des donnees de X : " << std::endl;
    for(int i = 0; i < rowsOfX; i++){
        for(int j = 0; j < colsOfX; j++){
            //std::cout << "X["<< i << "][" << j << "] =" << X[i] << std::endl;
            std::cout << "X["<< i << "] = " << X[i] << std::endl;
        }
    }

    std::cout << "Affichage des donnees de Y : " << std::endl;
    for(int i = 0; i < rowsOfY; i++){
        for(int j = 0; j < colsOfY; j++){
            //std::cout << "X["<< i << "][" << j << "] =" << X[i] << std::endl;
            std::cout << "Y["<< i << "] = " << Y[i] << std::endl;
        }
    }

    for(int i = 0; i < epochs; i++){

        //build phi
        for(int x = 0; x < rowsOfX; x++){
            for(int j = 0; j < colsOfX; j++){
                //phi[x][j] = exp(-gamma* (sqrt(X[x][j] * X[x][j] + X[x][j] * X[x][j])));
                phi[x][j] = exp(-gamma* (sqrt(X[x] * X[x] + X[x] * X[x])));
            }
        }

        double teta[rowsOfX][colsOfX];
        for(int i = 0; i < rowsOfX;i++){
            for(int j = 0; j < colsOfX;j++){
                //teta[i][j] = exp(-gamma * (sqrt(X[i][j] * X[i][j] + X[i][j] * X[i][j])));
                teta[i][j] = exp(-gamma * (sqrt(X[i] * X[i] + X[i] * X[i])));
                std::cout << "Teta[" << i <<"]"<<"["<<j<<"] = "<< teta[i][j] << std::endl;
            }
        }

        Matrix<double> tetaMatrix(rowsOfX, colsOfX);
        for (int i = 0; i < rowsOfX; ++i) {
            for (int j = 0; j < colsOfX; ++j) {
                tetaMatrix.put(i, j, teta[i][j]);
            }
        }

        Matrix<double> YMatrix(rowsOfY,colsOfY);

        //Dimensions de la matrice teta et de la matrice Y
        std::cout << "Dimensions de la matrice Teta : " << tetaMatrix.getRows() << "x" << tetaMatrix.getColumns() << std::endl;
        std::cout << "Dimensions de la matrice Y : " << YMatrix.getRows() << "x" << YMatrix.getColumns() << std::endl;


        // Matrix<double> wn = tetaMatrix.getInverse() * YMatrix;
        // //double res[wn.getRows()][wn.getColumns()];
        // for(int i = 0 ; i < wn.getRows(); i++){
        //     for(int j = 0; j < wn.getColumns(); j++){
        //         //res[i][j] = wn.get(i,j);
        //         std::cout << wn.get(i,j) << std::endl;
        //     }
        // }

        /*
        for(int i = 0; i < wn.getRows(); i++){
            for(int j = 0; j < wn.getColumns(); j++)
            std::cout << "Resultat : " << res[i][j] << std::endl;
        }
         */


        //double** teta = getTeta(gamma,X);
        //double wn = (reverse(getTeta) * YTrain) * exp(-gamma * norm(minus(XTrainM[i],XTrainM[epochs])));


        //exp(-gamma * norm(XTrain));
        //res +=
    }
    
}

}
