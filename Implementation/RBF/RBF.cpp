//
// Created by thuchard on 12/06/2019.
//

#if _WIN32
    #define SUPEREXPORT __declspec(dllexport)
    #include <ctime>
#else
    #define SUPEREXPORT
#endif

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "../Librairie/Eigen/Dense"
#include "../Librairie/Eigen/Eigen"

using namespace Eigen;
using namespace std;

extern "C" {

typedef struct RBF {
    MatrixXd W;
    MatrixXd X;
    MatrixXd MuMatrix;
    double gamma;
} RBF;

int getRand(int iter, int max) {
    srand(time(0) * (iter + 1 * 5));
    int size = max;
    int res = rand() % size;
    return res;
}

RBF *initRBF(MatrixXd X, MatrixXd WMatrix, double gamma, MatrixXd MuMatrix) {
    RBF *rbf = new RBF();
    rbf->W = WMatrix;
    rbf->X = X;
    rbf->gamma = gamma;
    rbf->MuMatrix = MuMatrix;
    return rbf;
}

SUPEREXPORT RBF *naive_rbf_train(double *X, double *Y, int inputCountPerSample, int sampleCount, double gamma = 100,
                     bool useBias = false) {
    MatrixXd XMatrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(X, inputCountPerSample, sampleCount);
    MatrixXd YMatrix = Map<MatrixXd>(Y, inputCountPerSample, 1);
    MatrixXd phi(inputCountPerSample, inputCountPerSample);
    for (int i = 0; i < inputCountPerSample; i++) {
        for (int j = 0; j < inputCountPerSample; j++) {
            phi(i, j) = exp(-gamma * (pow(((XMatrix.row(i) - XMatrix.row(j)).norm()), 2)));
        }
    }
    MatrixXd W = phi.inverse() * YMatrix;
    RBF *rbf = initRBF(XMatrix, W, gamma, XMatrix);
    return rbf;
}

SUPEREXPORT int sign(double pred){
    if(pred < 0){
        return 0;
    }else{
        return 1;
    }
}

SUPEREXPORT int naive_rbf_predict(RBF *rbf, double *sample) {
    MatrixXd sampleMatrix = Map<MatrixXd>(sample, 1, rbf->X.cols());
    MatrixXd gaussianOutputs(1, rbf->X.rows());
    for (int i = 0; i < rbf->X.rows(); i++) {
        gaussianOutputs(0, i) = exp((-rbf->gamma) * (pow(((rbf->X.row(i) - sampleMatrix).norm()), 2)));
    }
    return sign((double)(gaussianOutputs * rbf->W).sum());
}

SUPEREXPORT RBF *rbf_train(double *X, double *Y, int inputCountPerSample, int sampleCount, int epochs = 100, int k = 2,
               double gamma = 100,
               bool useBias = false) {
    MatrixXd XMatrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(X, inputCountPerSample, sampleCount);
    MatrixXd YMatrix = Map<MatrixXd>(Y, inputCountPerSample, 1);
    MatrixXd AMatrix = Map<MatrixXd>(XMatrix.data(), k, sampleCount);
    MatrixXd DistanceMatrix = Map<MatrixXd>(XMatrix.data(), inputCountPerSample, 1);

    //Randomize AMatrix
    for (int i = 0; i < k; i++) {
        AMatrix.row(i) = XMatrix.row(getRand(i, inputCountPerSample));
    }

    //Initialize distance matrix with -1
    for (int i = 0; i < DistanceMatrix.rows(); i++) {
        DistanceMatrix(i, 0) = -1;
    }

    MatrixXd TempMatrix = MatrixXd::Zero(inputCountPerSample, sampleCount);
    MatrixXd AveragesMatrix = Map<MatrixXd>(DistanceMatrix.data(), inputCountPerSample, 1);
    for (int x = 0; x < epochs; x++) {
        TempMatrix = AveragesMatrix; //TODO modify this assignation with Averages Matrix

        //Fill DistanceMatrix with distance of each point with the centers
        for (int i = 0; i < XMatrix.rows(); i++) {
            for (int j = 0; j < AMatrix.rows(); j++) {
                double distance = (AMatrix.row(j) - XMatrix.row(i)).norm();
                if (DistanceMatrix(i, 0) == -1 || distance < DistanceMatrix(i, 0)) {
                    DistanceMatrix(i, 0) = distance;
                    AveragesMatrix(i,0) = AMatrix(j,0);
                }
            }
        }

        if (AveragesMatrix == TempMatrix) {
            MatrixXd TempMatrix2 = MatrixXd::Zero(inputCountPerSample, sampleCount);
            for (int i = 0; i < XMatrix.rows(); i++) {
                for (int j = 0; j < DistanceMatrix.rows(); j++) {
                    for (int l = 0; l < XMatrix.cols(); l++) {
                        if (DistanceMatrix(i, 0) == 0) {
                            TempMatrix2(i, l) += XMatrix(i, l);
                        }
                    }
                }
            }

            MatrixXd phi(inputCountPerSample, inputCountPerSample);
            for (int i = 0; i < inputCountPerSample; i++) {
                for (int j = 0; j < inputCountPerSample; j++) {

                    phi(i, j) = exp(-gamma * (pow(((XMatrix.row(i) - TempMatrix2.row(j)).norm()), 2)));

                }
            }
            cout << phi << endl;
            MatrixXd W = (phi.transpose() * phi).inverse() * phi.transpose() * YMatrix;
            cout << W << endl;
            RBF *rbf = initRBF(XMatrix, W, gamma, TempMatrix2);
            return rbf;
        }
    }
    return nullptr;
}

SUPEREXPORT double rbf_predict(RBF *rbf, double *sample) {
    MatrixXd sampleMatrix = Map<MatrixXd>(sample, 1, rbf->X.cols());
    MatrixXd gaussianOutputs(1, rbf->X.rows());
    for (int i = 0; i < rbf->X.rows(); i++) {
        gaussianOutputs(0, i) = exp((-rbf->gamma) * (pow(((rbf->X.row(i) - rbf->MuMatrix.row(i)).norm()), 2)));
    }
    return (gaussianOutputs * rbf->W).sum();
}

int main() {
    //TODO ajouter sign à la classif naive rbf predict
    double gamma = 100;
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

    double sample[2] = {0.28093573, 0.36177096};

    RBF *rbfModel = naive_rbf_train(X, Y, 10, 2, 100);
    double res = naive_rbf_predict(rbfModel, sample);

    cout << res << endl;

    RBF *rbfModel2 = rbf_train(X, Y, inputCountPerSample, sampleCount, 100, 2, 100);
    double res2 = rbf_predict(rbfModel2, sample);
    cout << res2 << endl;

}

}