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

#include "../Librairie/Eigen/Dense"
#include "../Librairie/Eigen/Eigen"

using namespace Eigen;
using namespace std;

/**
 *
 *
 * X: [[0.13984698 0.41485388]
 [0.28093573 0.36177096]
 [0.25704393 0.97695092]
 [0.05471647 0.8640708 ]
 [0.91900274 0.95617945]
 [0.1753089  0.67689523]
 [0.25784674 0.12366917]
 [0.97495302 0.01277128]
 [0.08287882 0.94833339]
 [0.39418121 0.79789368]]
 Y: [0.46119306 0.78636786 0.2617359  0.25985246 0.28554652 0.57842217
 0.35202585 0.11248387 0.72196561 0.60782134]

 value after predict for this dataset = 0.7863678645709891

 */

extern "C" {

int main() {
    double gamma = 1;
    int inputCountPerSample = 10;
    int sampleCount = 2;

    double X[20] = {
            0.13984698, 0.41485388,
            0.28093573, 0.36177096,
            0.25704393, 0.97695092,
            0.05471647, 0.8640708,
            0.91900274, 0.95617945,
            0.1753089, 0.67689523,
            0.25784674, 0.12366917,
            0.97495302, 0.01277128,
            0.08287882, 0.94833339,
            0.39418121, 0.7978936
    };

    double Y[10] = {
            0.46119306,
            0.78636786,
            0.2617359,
            0.25985246,
            0.28554652,
            0.57842217,
            0.35202585,
            0.11248387,
            0.72196561,
            0.60782134
    };


    MatrixXd XMatrix = Map<MatrixXd>(X, inputCountPerSample, sampleCount);
    MatrixXd YMatrix = Map<MatrixXd>(Y, inputCountPerSample, 1);

    /*
    MatrixXd X(10, 2);
    X(0, 0) = 0.13984698;
    X(0, 1) = 0.41485388;
    X(1, 0) = 0.28093573;
    X(1, 1) = 0.36177096;
    X(2, 0) = 0.25704393;
    X(2, 1) = 0.97695092;
    X(3, 0) = 0.05471647;
    X(3, 1) = 0.8640708;
    X(4, 0) = 0.91900274;
    X(4, 1) = 0.95617945;
    X(5, 0) = 0.1753089;
    X(5, 1) = 0.67689523;
    X(6, 0) = 0.25784674;
    X(6, 1) = 0.12366917;
    X(7, 0) = 0.97495302;
    X(7, 1) = 0.01277128;
    X(8, 0) = 0.08287882;
    X(8, 1) = 0.94833339;
    X(9, 0) = 0.39418121;
    X(9, 1) = 0.79789368;
     */
    /*
    double X[10][2] = {
            {0.13984698, 0.41485388},
            {0.28093573, 0.36177096},
            {0.25704393, 0.97695092},
            {0.05471647, 0.8640708},
            {0.91900274, 0.95617945},
            {0.1753089,  0.67689523},
            {0.25784674, 0.12366917},
            {0.97495302, 0.01277128},
            {0.08287882, 0.94833339},
            {0.39418121, 0.79789368}
    };

    */

    /*
    MatrixXd Y(10, 1);
    Y(0, 0) = 0.46119306;
    Y(1, 0) = 0.78636786;
    Y(2, 0) = 0.2617359;
    Y(3, 0) = 0.25985246;
    Y(4, 0) = 0.28554652;
    Y(5, 0) = 0.57842217;
    Y(6, 0) = 0.35202585;
    Y(7, 0) = 0.11248387;
    Y(8, 0) = 0.72196561;
    Y(9, 0) = 0.60782134;
    */

    /*
    double Y[10] = {
            0.46119306, 0.78636786, 0.2617359, 0.25985246, 0.28554652, 0.57842217,
            0.35202585, 0.11248387, 0.72196561, 0.60782134
    };*/
    MatrixXd phi(inputCountPerSample, inputCountPerSample);
    printf("Dimensions de X : %d x %d\n Dimensions de Y : %d x %d\n", XMatrix.rows(), XMatrix.cols(), YMatrix.rows(),
           YMatrix.cols());
    /*
    cout << "Nombre de colonnes pour X : " << rowsOfX << endl;
    cout << "Nombre de lignes pour X : " << colsOfX << endl;
    cout << "Nombre de colonnes pour Y : " << rowsOfY << endl;
    cout << "Nombre de lignes pour Y : " << colsOfY << endl;
    cout << "Affichage des donnees de X : " << X << endl;
    */

    /*
    cout << "Affichage des donnees de X : " << endl;
    for (int i = 0; i < rowsOfX; i++) {
        for (int j = 0; j < colsOfX; j++) {
            cout << "X[" << i << "][" << j << "] = " << X[i][j] << endl;
        }
    }
    */
    //cout << "Affichage des donnees de X : " << XMatrix << endl;
    //cout << "Affichage des donnees de Y : " << YMatrix << endl;
    /*
    cout << "Affichage des donnees de Y : " << endl;
    for (int i = 0; i < rowsOfY; i++) {
        for (int j = 0; j < colsOfY; j++) {
            cout << "Y[" << i << "] = " << Y[i] << endl;
        }
    }
    */

    //build phi

    for (int i = 0; i < inputCountPerSample; i++) {
        for (int j = 0; j < inputCountPerSample; j++) {
            phi(i, j) = exp(-gamma * pow((XMatrix.row(i) - XMatrix.row(j)).norm(), 2));
            //cout << phi(i, j) << endl;
        }
    }

    double phiDoubleArray[20];
    int x = 0;
    for(int i = 0; i < inputCountPerSample; i++){
        for(int j = 0; j < inputCountPerSample; j++){
            phiDoubleArray[x] = phi(i,j);
            x++;
        }
    }

    MatrixXd W = Map<MatrixXd>(phiDoubleArray, inputCountPerSample, inputCountPerSample);

    cout << W << endl;

    //MatrixXd W = phi.inverse() * YMatrix;

    //cout << W << endl;

    //Dimensions de la matrice teta et de la matrice Y
    //cout << "Dimensions de la matrice Teta : " << phi.rows() << "x"
    //     << phi.cols()
    //    << endl;
    //cout << "Dimensions de la matrice Y : " << YMatrix.rows() << "x"
    //     << YMatrix.cols()
    //    << endl;

    // Matrix<double> wn = tetaMatrix.getInverse() * YMatrix;
    // //double res[wn.getRows()][wn.getColumns()];
    // for(int i = 0 ; i < wn.getRows(); i++){
    //     for(int j = 0; j < wn.getColumns(); j++){
    //         //res[i][j] = wn.get(i,j);
    //         cout << wn.get(i,j) << endl;
    //     }
    // }

    /*
    for(int i = 0; i < wn.getRows(); i++){
        for(int j = 0; j < wn.getColumns(); j++)
        cout << "Resultat : " << res[i][j] << endl;
    }
     */

    //double** teta = getTeta(gamma,X);
    //double wn = (reverse(getTeta) * YTrain) * exp(-gamma * norm(minus(XTrainM[i],XTrainM[epochs])));

    //exp(-gamma * norm(XTrain));
}


}
