//
// Created by Baptiste Vasseur on 2019-05-18.
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

#include "../Librairie/Matrix.h"

//extern "C" {

struct MLP{
    int* npl;
    int layer_count;
    double*** W;
    double** X;
    double** deltas;
};

//IMPORT FROM ROSENBLATT ALGORITHM

SUPEREXPORT double getRand(double min, double max) {

    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);

    return val;
}

SUPEREXPORT double* create_linear_model(int inputCountPerSample) {
    auto res = new double[inputCountPerSample + 1];

    for (int i = 0; i < inputCountPerSample + 1; i++) {
        res[i] = getRand(-1.0, 1.0);
    }

    return res;
}


SUPEREXPORT double** bias(int ligne, int col){
    auto XTrain = new double*[ligne];

    for (int i = 0; i < ligne; ++i) {
        XTrain[i] = new double[col];
        XTrain[i][0] = 1.0;
    }
    return XTrain;
}

SUPEREXPORT double** oneDimToTwo(const double* XTrain, int ligne, int col){
    int pos = 0;
    double** XTrainFin = bias(ligne, col);

    for (int i = 0; i < ligne; i++) {
        XTrainFin[i] = new double[col];

        for (int j = 0; j < col; j++) {
            XTrainFin[i][j] = XTrain[pos];
            pos += 1;
        }
    }

    return XTrainFin;
}

//MLP ALGORITHM

void init_model(MLP* mlp){
    mlp->W = new double**[mlp->layer_count];

    for (int l = 1; l < mlp->layer_count; ++l) {
        int prev_neuron_count = mlp->npl[l - 1] + 1; //+1 pour le biais
        int cur_neuron_count = mlp->npl[l] + 1;

        mlp->W[l] = new double*[prev_neuron_count];

        for (int j = 0; j < prev_neuron_count; ++j) {
            mlp->W[l][j] = new double[cur_neuron_count];
            mlp->W[l][j] = create_linear_model(cur_neuron_count);
        }
    }
}

void feedFoward(MLP* mlp){

    /**
     * feedFoward
     */

    for (int l = 1; l < mlp->layer_count; l++) {

        mlp->X[l] = new double[mlp->npl[l] + 1];
        mlp->X[l][0] = 1;

        int prev_neuron_count = mlp->npl[l - 1] + 1; //+1 pour le biais
        int cur_neuron_count = mlp->npl[l] + 1;

        for (int j = 1; j < cur_neuron_count; j++) { //+1 pour le biais
            double val = 0;

            for (int i = 0; i < prev_neuron_count; i++) {
                val += (mlp->W[l][i][j] * mlp->X[l - 1][i]);
            }

            mlp->X[l][j] = tanh(val);
        }
    }
}



double sommWxDelta(const double* w, int cur_neuron_count, double* delta){
    double res = 0;
    for (int j = 1; j < cur_neuron_count; j++) {
        res += w[j] * delta[j];
    }
    return res;
}

void initAllDeltaExpectLast(MLP* mlp, int lastIndex){
    for (int l = lastIndex; l > 0 ; l--) {
        int prev_neuron_count = mlp->npl[l - 1] + 1;
        int cur_neuron_count = mlp->npl[l] + 1;
        mlp->deltas[l - 1] = new double[prev_neuron_count];
        for (int i = 1; i < prev_neuron_count; i++) {
            double som = sommWxDelta(mlp->W[l][i], cur_neuron_count, mlp->deltas[l]);
            double val = (1 - pow(mlp->X[l - 1][i], 2)) * som;
            mlp->deltas[l - 1][i] = val;
        }
    }
}

void initLastDelta_classification(MLP* mlp, int lastIndex, const double* YTrain){
    mlp->deltas[lastIndex] = new double[mlp->npl[lastIndex] + 1];
    for (int j = 1; j < mlp->npl[lastIndex] + 1; j++) {
        double val = (1 - pow((mlp->X[lastIndex][j]),2)) * (mlp->X[lastIndex][j] - YTrain[j - 1]);
        mlp->deltas[lastIndex][j] = val;
    }
}

void initLastDelta_regression(MLP* mlp, int lastIndex, const double* YTrain){
    mlp->deltas[lastIndex] = new double[mlp->npl[lastIndex] + 1];
    for (int j = 1; j < mlp->npl[lastIndex] + 1; j++) {
        double val = (mlp->X[lastIndex][j] - YTrain[j - 1]);
        mlp->deltas[lastIndex][j] = val;
    }
}

void updateW(MLP* mlp, int lastIndex, double alpha){
    for (int l = 1; l < lastIndex + 1; l++) {
        int prev_neuron_count = mlp->npl[l - 1] + 1;
        int cur_neuron_count = mlp->npl[l] + 1;

        for (int j = 1; j < cur_neuron_count; j++) {
            for (int i = 0; i < prev_neuron_count; i++) {
                mlp->W[l][i][j] = mlp->W[l][i][j] - (alpha * mlp->X[l - 1][i] * mlp->deltas[l][j]);
            }
        }
    }
}

void displayAllWValues(MLP* mlp){
    //PRINT ALL W
    for (int l = 1; l < mlp->layer_count; ++l) {
        int prev_neuron_count = mlp->npl[l - 1] + 1; // +1 pour le biais
        int cur_neuron_count = mlp->npl[l] + 1;
        for (int j = 0; j < prev_neuron_count; ++j) { // +1 pour le biais
            for (int k = 1; k < cur_neuron_count; ++k) {
                std::cout << "w[" << l << "][" << j << "]["  << k << "] : " <<  mlp->W[l][j][k] << " \n";
            }
        }
        std::cout << std::endl;
    }
}

void displayAllXValues(MLP* mlp){
    //PRINT ALL X
    for (int l = 0; l < mlp->layer_count; ++l) {
//        std::cout << "x[" << l << "] : ";
        for (int j = 0; j < mlp->npl[l] + 1; ++j) {
//            std::cout << mlp->X[l][j] << " - ";
            std::cout << "x[" << l << "][" << j << "] : " <<  mlp->X[l][j] << " \n";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void displayAllDeltaValues(MLP* mlp){
    //PRINT ALL Deltas
    for (int l = 0; l < mlp->layer_count; ++l) {
//    std::cout << "delta[" << l << "] : ";
        for (int j = 0; j < mlp->npl[l] + 1; ++j) {
//            std::cout << mlp->deltas[l][j] << " - ";
            std::cout << "delta[" << l << "][" << j << "] : " <<  mlp->deltas[l][j] << " \n";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {

    srand(time(nullptr));

    int sampleCount = 4;
    int inputCountPerSample = 2;
    double alpha = 0.001;
    int epoch = 10000;

    double XTrain[8] = {
            0, 0,
            1, 0,
            0, 1,
            1, 1
    };

    double YTrain[12] = {
            1, -1, -1,
            -1, 1, -1,
            -1, 1, -1,
            -1, -1, 1
    };

    MLP m;
    MLP* mlp;
    mlp = &m;

    mlp->layer_count = 4;

    int NPL[4] = {2, 2,2, 3};
    mlp->npl = NPL;

    mlp->deltas = new double*[mlp->layer_count];
    auto** x = new double*[mlp->layer_count];
    mlp->X = x;

    init_model(mlp);

    auto x2 = oneDimToTwo(XTrain, sampleCount, inputCountPerSample + 1);
    auto y2 = oneDimToTwo(YTrain, sampleCount, 3);

    int lastLayerIndex = mlp->layer_count - 1;

//    displayAllWValues(mlp);

    for (int e = 0; e < epoch; ++e) {
        for (int i = 0; i < sampleCount; ++i) {

            mlp->X[0] = x2[i];
            //cout << "--> Feed Foward" << std::endl;
            feedFoward(mlp);
            //cout << "--> Back Propagation" << std::endl;

            initLastDelta_classification(mlp, lastLayerIndex, y2[i]);
            initAllDeltaExpectLast(mlp, lastLayerIndex);
            updateW(mlp, lastLayerIndex, alpha);
        }
    }

//    std::cout << "-------------------" << std::endl;
//    displayAllWValues(mlp);

    double XtoPred1[3] = {1, 0, 0};
    mlp->X[0] = XtoPred1;
    feedFoward(mlp);
    displayAllXValues(mlp);

    double XtoPred2[3] = {1, 0, 1};
    mlp->X[0] = XtoPred2;
    feedFoward(mlp);
    displayAllXValues(mlp);

    double XtoPred3[3] = {1, 1, 0};
    mlp->X[0] = XtoPred3;
    feedFoward(mlp);
    displayAllXValues(mlp);

    double XtoPred4[3] = {1, 1, 1};
    mlp->X[0] = XtoPred4;
    feedFoward(mlp);
    displayAllXValues(mlp);
}


//}