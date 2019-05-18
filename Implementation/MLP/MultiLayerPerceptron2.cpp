//
// Created by Théo Huchard on 15/05/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "../Librairie/Matrix.h"

double getRand(double min, double max) {
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);
    return val;
}

double *initW(int layers, int XTrainRows, int XTrainColumns) {

    auto *res = new double[layers * XTrainRows * XTrainColumns];
    //std::cout << layers * XTrainRows * XTrainColumns << std::endl;
    for (int k = 0; k < layers * XTrainRows * XTrainColumns; k++) {
        res[k] = getRand(-1, 1);
        //std::cout << res[k] << std::endl;
    }

    return res;
}

double *sigmoid_sum(double *XTrain, double *W, double DL, int j) {
    double *result = nullptr;
    for (int i = 0; i < DL; i++) {
        W[i * j * ] =
    }
}

void fit_classification(double *XTrain, double *YTrain, int XTrainRows, int XTrainColumns, int YTrainRows,
                        int epochs) {
    double *W = initW(epochs, XTrainRows, XTrainColumns);
    //Classification
    for (int i = 0; i < epochs; i++) {
        double *xlj = tanh(sigmoid_sum(XTrain, W, XTrainRows,i));
        double **deltaLastLayer = product(
                difference(1, square(XTrain, 3, 4)),
                difference(XTrain, YTrain, XTrainRows, XTrainColumns, YTrainRows)
        );
        double **deltaAllExceptLastLayer = product(
                difference(1 - square(XTrain))
        );
    }
}

int main() {

    srand(time(nullptr)); // Enable rand() function

    double XTrain[]{
            4.3, 3.2, 1.4,
            4.9, 3.0, 1.4,
            6.2, 3.4, 5.4,
            5.9, 3.0, 5.1
    };
    double YTrain[]{-1, 1, 1};
    int epochs(50);

    //Classification
    fit_classification(XTrain, YTrain, 3, 4, 1, epochs);

    //Regression
}
