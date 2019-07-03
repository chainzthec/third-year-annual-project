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
#include <ctime>

struct MLP{
    int* npl; // Neuron per layers
    int layer_count; // Number of layer
    double*** W; // All W values
    double** X; // All X values
    double** deltas; // All delta values
    int lastLayerIndex;
};

extern "C" {

//IMPORT FROM ROSENBLATT ALGORITHM

SUPEREXPORT double getRand(double min, double max) {

    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);

    return val;
}

SUPEREXPORT double* init_with_random(int inputCountPerSample) {
    auto res = new double[inputCountPerSample + 1];

    for (int i = 0; i < inputCountPerSample + 1; i++) {
        res[i] = getRand(-1.0, 1.0);
//        res[i] = 0.5;
    }

    return res;
}

SUPEREXPORT double** addMatrixBias(double** XtoPred1, int sampleCount, int inputCountPerSample){

    auto* res = new double*[sampleCount];

    for (int j = 0; j < sampleCount; j++) {
        res[j] = new double[inputCountPerSample + 1];
        res[j][0] = 1.0;

        for (int i = 0; i < inputCountPerSample; i++) {
            res[j][i+1] = XtoPred1[j][i];
        }
    }

    return res;
}

SUPEREXPORT double* addBias(const double* XLineToPred, int inputCountPerSample){
    auto res = new double[inputCountPerSample + 1];

    res[0] = 1.0;

    for (int j = 0; j < inputCountPerSample; j++) {
        res[j+1] = XLineToPred[j];
    }

    return res;
}

SUPEREXPORT double** convertToMatrix(const double* XTrain, int ligne, int col){
    int pos = 0;
    auto** XTrainFin = new double*[ligne];

    for (int i = 0; i < ligne; i++) {
        XTrainFin[i] = new double[col];

        for (int j = 0; j < col; j++) {
            XTrainFin[i][j] = XTrain[pos];
            pos += 1;
        }
    }

    return XTrainFin;
}

// MLP ALGORITHM

SUPEREXPORT void init_model(MLP* mlp){

    mlp->W = new double**[mlp->layer_count];

    for (int l = 1; l < mlp->layer_count; l++) {
        int prev_neuron_count = mlp->npl[l - 1] + 1; // +1 pour le biais
        int cur_neuron_count = mlp->npl[l] + 1; // +1 pour le biais

        mlp->W[l] = new double*[prev_neuron_count];

        for (int j = 0; j < prev_neuron_count; j++) { // HERE
            mlp->W[l][j] = new double[cur_neuron_count];
            mlp->W[l][j] = init_with_random(cur_neuron_count);
        }
    }
}

SUPEREXPORT MLP* create_MLP() {
    auto* ret = (struct MLP*) malloc(sizeof(struct MLP));
    ret->X = new double*;
    ret->npl = new int;
    ret->lastLayerIndex = 0;
    ret->deltas = new double*;
    ret->layer_count = 0;
    ret->W = new double**;

    return ret;
}

SUPEREXPORT MLP* init(const int* neurons, int size){

    srand(time(nullptr));

    MLP* mlp = create_MLP();

    mlp->layer_count = size;
    mlp->npl = (int*) malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        mlp->npl[i] = neurons[i];
    }

    mlp->X = new double*[size];
    mlp->deltas = new double*[size];

    init_model(mlp);

    mlp->lastLayerIndex = mlp->layer_count - 1;

    return mlp;
}

SUPEREXPORT void feedFoward(MLP* mlp){
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

SUPEREXPORT double sommWxDelta(const double* w, int cur_neuron_count, const double* delta){
    double res = 0;
    for (int j = 1; j < cur_neuron_count; j++) {
        res += w[j] * delta[j];
    }
    return res;
}

SUPEREXPORT void initAllDeltaExceptLast(MLP* mlp, int lastIndex){
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

SUPEREXPORT void initLastDelta_classification(MLP* mlp, int lastIndex, const double* YTrain){
    mlp->deltas[lastIndex] = new double[mlp->npl[lastIndex] + 1];
    for (int j = 1; j < mlp->npl[lastIndex] + 1; j++) {
        double val = (1 - pow((mlp->X[lastIndex][j]),2)) * (mlp->X[lastIndex][j] - YTrain[j - 1]);
        mlp->deltas[lastIndex][j] = val;
    }
}

SUPEREXPORT void initLastDelta_regression(MLP* mlp, int lastIndex, const double* YTrain){
    mlp->deltas[lastIndex] = new double[mlp->npl[lastIndex] + 1];
    for (int j = 1; j < mlp->npl[lastIndex] + 1; j++) {
        double val = (mlp->X[lastIndex][j] - YTrain[j - 1]);
        mlp->deltas[lastIndex][j] = val;
    }
}

SUPEREXPORT void updateW(MLP* mlp, int lastIndex, double alpha){
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

<<<<<<< HEAD
SUPEREXPORT void destroy(struct MLP* mlp) {
    if( mlp ){
        free( mlp );
    }
}

SUPEREXPORT double* predict(MLP* mlp, double* XtoPred){
//    std::cout << "------- Test -------" << std::endl;

    int inputCountPerSample = mlp->npl[0];
    double* XtoPredWithBias = addBias(XtoPred, inputCountPerSample);
    mlp->X[0] = XtoPredWithBias;
    feedFoward(mlp);

    int last_layer_neuron_count = mlp->npl[mlp->lastLayerIndex];
    auto* res = new double[last_layer_neuron_count];

    for (int i = 1; i < last_layer_neuron_count + 1; i++) {
        auto value = mlp->X[mlp->lastLayerIndex][i];
        res[i-1] = value ;
    }

    return res;
}

SUPEREXPORT MLP* fit_classification(MLP* mlp, double* XTrain, double* YTrain, int sampleCount, int epochs, double alpha) {

    int startNeuron = mlp->npl[0];
    int endNeuron = mlp->npl[mlp->lastLayerIndex];
=======
//SUPEREXPORT void displayAllWValues(MLP* mlp){
//    //  PRINT ALL W
//    for (int l = 1; l < mlp->layer_count; l++) {
//        int prev_neuron_count = mlp->npl[l - 1] + 1; // +1 pour le biais
//        int cur_neuron_count = mlp->npl[l] + 1;
//        for (int j = 0; j < prev_neuron_count; j++) { // +1 pour le biais
//            for (int k = 1; k < cur_neuron_count; k++) {
//                std::cout << "w[" << l << "][" << j << "]["  << k << "] : " <<  mlp->W[l][j][k] << " \n";
//            }
//        }
//        std::cout << std::endl;
//    }
//}
//
//SUPEREXPORT void displayAllXValues(MLP* mlp){
//    //  PRINT ALL X
//    for (int l = 0; l < mlp->layer_count; l++) {
////        std::cout << "x[" << l << "] : ";
//        for (int j = 0; j < mlp->npl[l] + 1; j++) {
////            std::cout << mlp->X[l][j] << " - ";
//            std::cout << "x[" << l << "][" << j << "] : " <<  mlp->X[l][j] << " \n";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}
//
//SUPEREXPORT void displayAllDeltaValues(MLP* mlp){
//    //  PRINT ALL Deltas
//    for (int l = 0; l < mlp->layer_count; l++) {
////    std::cout << "delta[" << l << "] : ";
//        for (int j = 0; j < mlp->npl[l] + 1; j++) {
////            std::cout << mlp->deltas[l][j] << " - ";
//            std::cout << "delta[" << l << "][" << j << "] : " <<  mlp->deltas[l][j] << " \n";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}

SUPEREXPORT double* predict(MLP* mlp, double* XtoPred){
//    std::cout << "------- Test -------" << std::endl;

    int inputCountPerSample = mlp->npl[0];
    double* XtoPredWithBias = addBias(XtoPred, inputCountPerSample);
    mlp->X[0] = XtoPredWithBias;
    feedFoward(mlp);

    int last_layer_neuron_count = mlp->npl[mlp->lastLayerIndex];
    auto* res = new double[last_layer_neuron_count];

    for (int i = 1; i < last_layer_neuron_count + 1; i++) {
        auto value = mlp->X[mlp->lastLayerIndex][i];
//        res[i] = mlp->X[mlp->lastLayerIndex][i+1];
//        res[i] = value < 0 ? -1 : 1 ;
        res[i-1] = value ;
    }

    return res;
}

SUPEREXPORT MLP* create_MLP() {
    auto* ret = (struct MLP*) malloc(sizeof(struct MLP));
    ret->X = new double*;
    ret->npl = new int;
    ret->lastLayerIndex = 0;
    ret->deltas = new double*;
    ret->layer_count = 0;
    ret->W = new double**;

    return ret;
}

SUPEREXPORT MLP* init(const int* neurons, int size){

    srand(time(nullptr));

    MLP* mlp = create_MLP();

    mlp->layer_count = size;
    mlp->npl = (int*) malloc(sizeof(int) * size);

    for (int i = 0; i < size; i++) {
        mlp->npl[i] = neurons[i];
    }
>>>>>>> master

    double** XTrainFinal = convertToMatrix(XTrain, sampleCount, startNeuron);
    XTrainFinal = addMatrixBias(XTrainFinal, sampleCount, startNeuron);

<<<<<<< HEAD
    double** YTrainFinal = convertToMatrix(YTrain, sampleCount, endNeuron);


    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < sampleCount; i++) {

            mlp->X[0] = XTrainFinal[i];
            feedFoward(mlp);

            initLastDelta_classification(mlp, mlp->lastLayerIndex, YTrainFinal[i]);
            initAllDeltaExceptLast(mlp, mlp->lastLayerIndex);
            updateW(mlp, mlp->lastLayerIndex, alpha);
        }
    }
=======
    init_model(mlp);

    mlp->lastLayerIndex = mlp->layer_count - 1;
>>>>>>> master

    return mlp;
}

<<<<<<< HEAD
SUPEREXPORT MLP* fit_regression(MLP* mlp, double* XTrain, double* YTrain, int sampleCount, int epochs, double alpha) {
=======
SUPEREXPORT MLP* fit_classification(MLP* mlp, double* XTrain, double* YTrain, int sampleCount, int epochs, double alpha) {
>>>>>>> master

    int startNeuron = mlp->npl[0];
    int endNeuron = mlp->npl[mlp->lastLayerIndex - 1];

    double** XTrainFinal = convertToMatrix(XTrain, sampleCount, startNeuron);
    XTrainFinal = addMatrixBias(XTrainFinal, sampleCount, endNeuron);
<<<<<<< HEAD

    double** YTrainFinal = convertToMatrix(YTrain, sampleCount, 1);
=======

    double** YTrainFinal = convertToMatrix(YTrain, sampleCount, 1);

>>>>>>> master

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < sampleCount; i++) {

            mlp->X[0] = XTrainFinal[i];
            feedFoward(mlp);

<<<<<<< HEAD
            initLastDelta_regression(mlp, mlp->lastLayerIndex, YTrainFinal[i]);
=======
            initLastDelta_classification(mlp, mlp->lastLayerIndex, YTrainFinal[i]);
>>>>>>> master
            initAllDeltaExceptLast(mlp, mlp->lastLayerIndex);
            updateW(mlp, mlp->lastLayerIndex, alpha);
        }
    }

    return mlp;
<<<<<<< HEAD
=======
}

SUPEREXPORT MLP* fit_regression(MLP* mlp, double* XTrain, double* YTrain, int sampleCount, int epochs, double alpha) {

    int startNeuron = mlp->npl[0];
    int endNeuron = mlp->npl[mlp->lastLayerIndex - 1];

    double** XTrainFinal = convertToMatrix(XTrain, sampleCount, startNeuron);
    XTrainFinal = addMatrixBias(XTrainFinal, sampleCount, endNeuron);

    double** YTrainFinal = convertToMatrix(YTrain, sampleCount, 1);

    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < sampleCount; i++) {

            mlp->X[0] = XTrainFinal[i];
            feedFoward(mlp);

            initLastDelta_regression(mlp, mlp->lastLayerIndex, YTrainFinal[i]);
            initAllDeltaExceptLast(mlp, mlp->lastLayerIndex);
            updateW(mlp, mlp->lastLayerIndex, alpha);
        }
    }

    return mlp;
}

SUPEREXPORT void destroy(struct MLP* mlp) {
    if( mlp ){
        free( mlp );
    }
>>>>>>> master
}

int main() {

    // Init

    int sampleCount = 4;
    double alpha = 0.001;
    int epochs = 50000;

    double XTrain[8] = {
            0, 0,
            1, 0,
            0, 1,
            1, 1
    };
    double YTrain[4] = {1, -1, -1, 1};

    // MLP implemtation

    int neurons[3] = {2, 2, 1};
    MLP* mlp = init(neurons, 3);

<<<<<<< HEAD
    mlp = fit_classification(mlp, XTrain, YTrain, sampleCount, epochs, alpha);

//    double XtoPred1[2] = {0, 0};
//    auto* res = predict(mlp, XtoPred1);
//
//    for (int i = 0; i < 3; ++i) {
//        std::cout << res[i] << " ";
//    }

//    double XtoPred2[2] = {1, 0};
//    predict(mlp, XtoPred2);
//
//    double XtoPred3[2] = {0, 1};
//    predict(mlp, XtoPred3);
//
//    double XtoPred4[2] = {1, 1};
//    predict(mlp, XtoPred4);
=======
    mlp = fit_regression(mlp, XTrain, YTrain, sampleCount, epochs, alpha);

    double XtoPred1[2] = {0, 0};
    predict(mlp, XtoPred1);

    double XtoPred2[2] = {1, 0};
    predict(mlp, XtoPred2);

    double XtoPred3[2] = {0, 1};
    predict(mlp, XtoPred3);

    double XtoPred4[2] = {1, 1};
    predict(mlp, XtoPred4);
>>>>>>> master

//    std::cout << "-------------------" << std::endl;
//    displayAllWValues(mlp);

<<<<<<< HEAD
//    std::cout << "\n";
//
=======
>>>>>>> master
//    for (double i = 1; i >= -0.05; i-=0.05) {
//
//        printf("%4.2f > ", i);
//        for (double j = 0; j <= 1.05; j+=0.05) {
//
<<<<<<< HEAD
//            double XtoPred[2] = {i, j};
//            feedFoward(mlp);
//
//            auto* values = predict(mlp, XtoPred);
//
//            std::cout << " " << values[0];
//
//            if( values[0] > 0 )
=======
//            double XtoPred[3] = {1, i, j};
//            (mlp)->X[0] = XtoPred;
//            feedFoward(mlp);
//
//            if( (mlp)->X[( (mlp)->lastLayerIndex )][1] > 0 )
>>>>>>> master
//                std::cout << " x ";
//            else
//                std::cout << " - ";
//
//        }
//        std::cout << "\n";
//    }
<<<<<<< HEAD
//
//    destroy(mlp);
=======

    destroy(mlp);
>>>>>>> master
}


}