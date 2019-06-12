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

SUPEREXPORT double getNormAbsValueOfXMinusXN(double *XTrain, double rows, double columns){
    double sum = 0;
    auto* phi = new double[rows * columns];
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            phi[i+j] = abs(XTrain[i] - XTrain[j]) * abs(XTrain[i] - XTrain[j]);
            sum += phi[i+j];
        }
    }
    return sum;
}

SUPEREXPORT double* fit_classification(
        double* XTrain,
        int sampleCount, // nombre d'image (ligne)
        int inputCountPerSample, //nombre de pixel par img (colonne)
        double gamma,
        int epochs // Nombre d'itération
) {
    double currentWeight;
    //somme de 1 à N
    for(int i = 0; i < epochs; i++){
        double absValueOfXMinusXN = exp((-gamma) * getNormAbsValueOfXMinusXN(XTrain, sampleCount, inputCountPerSample));
        std::cout << "Exp value = " << absValueOfXMinusXN << std::endl;
        //double* valueOfExponential = ;
        //currentWeight = calcWeight(W);
    }

    return nullptr;
}

int main() {

    srand(time(nullptr)); // Enable rand() function

    int inputCountPerSample = 2;
    int sampleCount = 13;
    int epochs = 10;
    double gamma = 0.01;

    double Xtrains[26] = {
            0, 0,
            1, 0,
            0, 1,
            2, 2,
            1, 2,
            2, 1,
            0.25, 0.25,
            0.1, 0.1,
            0.15, 0.15,
            0.3, 0.3,
            3, 3,
            1.5, 1.5,
            2.5, 2.5
    };

    // 13 car sampleCount = 13 (soit 13 images)
    //double Ytrains[13] = {-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1};

    double * res = fit_classification(Xtrains,sampleCount,inputCountPerSample,gamma, epochs);



}

}
