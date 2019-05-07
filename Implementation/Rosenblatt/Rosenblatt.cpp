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

/*
 *
 * Utils
 *
 */

// ajouter une autre fonction sign(somme de 0 à sampleCount) (w_i * x_i) ** 0
// bias = position de la droite, a et b orientation de la droite

SUPEREXPORT double getRand(double fMin, double fMax) {
//    double val = rand() % 2; // rand() need to be enabled
//    val = (val == 0) ? -1.0 : 1.0 ;

    double val = (double) rand() / RAND_MAX;
    val = fMin + val * (fMax - fMin);

//    std::cout << "Rand : " << val << std::endl;
    return val;
}

SUPEREXPORT Matrix<double> create_linear_model(int inputCountPerSample) {
    Matrix<double> matrix(1, inputCountPerSample);

    for (int i = 0; i < inputCountPerSample; ++i) {
        double r = getRand(-1.0, 1.0);
        matrix.put(0, i, r);
    }

    return matrix;
}

SUPEREXPORT Matrix<double> addBias(const Matrix<double>& matrix){

    int bias = 1;
    Matrix<double> result(matrix.getRows(), matrix.getColumns() + 1);

    for (int i = 0; i < matrix.getRows(); ++i) {
        result.put(i, 0, bias);
        for (int j = 0; j < matrix.getColumns(); ++j) {
            result.put(i, j+1, matrix.get(i, j));
        }
    }

    return result;
}

/*
 *
 * Implementation
 *
 */

SUPEREXPORT double predict_regression(
        const Matrix<double>& W,
        const Matrix<double>& XLineToPredict
) {
    // TODO : Inférence (CF Slides !)

    double res = 0.0;
    for (int j = 0; j < XLineToPredict.getColumns(); ++j) {
        res = res + (W.get(0, j) * XLineToPredict.get(0, j));
    }

    res += W.get(0, 0);
    return res;
}

SUPEREXPORT double predict_classification(
        const Matrix<double>& W,
        const Matrix<double>& XLineToPredict
) {
    return predict_regression(W, XLineToPredict) >= 0 ? 1.0 : -1.0;
}

Matrix<double> fit_regression(
        const Matrix<double>& W,
        const Matrix<double>& XTrain,
        const Matrix<double>& YTrain
) {
    // TODO : entrainement (correction des W, cf slides !)

//    W = ( (transpose(XTrain) * XTrain)^-1 * transpose(XTrain) * YTrain
    Matrix<double> XTrainTranspose = XTrain.getTranspose();
    Matrix<double> XTmultX = XTrainTranspose * XTrain;
    Matrix<double> XTmultXInverse = XTmultX.getInverse();

    return XTmultXInverse * XTrainTranspose * YTrain;
}


SUPEREXPORT Matrix<double> fit_classification(
        Matrix<double> W,
        const Matrix<double>& XTrain,
        const Matrix<double>& YTrain,
        double alpha, // Learning Rate (0,01 par exemple)
        int epochs // Nombre d'itération
) {
    Matrix<double> XTrainFin = addBias(XTrain);

    int sampleCount = XTrain.getRows(); // Nombres images

    for (auto i = 0; i < epochs; i++) {
        for (auto j = 0; j < sampleCount; j++) {

            //TODO : entrainement (correction des W, cf slides !)
            Matrix<double> tmp = XTrainFin.getRow(j);

            for (auto k = 0; k < XTrainFin.getColumns(); k++) {
                double predict = predict_classification(W, tmp);
                double value = W.get(0, k) + alpha * (YTrain.get(j, 0) - predict) * XTrainFin.get(j, k);
                W.put(0, k, value);
            }
        }
    }

    return W;
}

SUPEREXPORT void delete_linear_model(double *W) {
    delete[] W;
}






SUPEREXPORT Matrix<double> transformDoubleToMatrix(double* mat, int rows, int cols = 1){
    int currentPos = 0;
    Matrix<double> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.put(i, j, mat[currentPos] );
            currentPos += 1;
        }
    }

    return res;
}


int main() {

    // TODO : main & test

    srand(time(nullptr)); // Enable rand() function

    int inputCountPerSample = 2;
    int sampleCount = 13;
    int epochs = 5000;
    double alpha = 0.001;

    Matrix<double> model = create_linear_model(inputCountPerSample);
    model = addBias(model);

//    std::cout << "model: " << std::endl << model << std::endl;
//
    // 26 car sampleCount * inputCountPerSample = 13 * 2 = 26 (soit 13 images de 2 pixels donc 26 pixels au total)
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
    double Ytrains[13] = {-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1};

    Matrix<double> XtrainsMatrix = transformDoubleToMatrix(Xtrains, sampleCount, inputCountPerSample);
    Matrix<double> YtrainsMatrix = transformDoubleToMatrix(Ytrains, sampleCount, 1);

    /*
     * Classification
     */

//    std::cout <<  "Before Rosenblatt : " << std::endl;
//    std::cout << model << std::endl;
//    model = fit_classification(model, XtrainsMatrix, YtrainsMatrix, alpha, epochs);
//    std::cout <<  "After Rosenblatt : " << std::endl;
//    std::cout << model << std::endl;
//
//    double val1[2] = {0.1, 0.1};
//    Matrix<double> XtoPred1 = transformDoubleToMatrix(val1, 1, inputCountPerSample);
//
//    double val2[2] = {2, 2};
//    Matrix<double> XtoPred2 = transformDoubleToMatrix(val2, 1, inputCountPerSample);
//
//    double val3[2] = {3, 3};
//    Matrix<double> XtoPred3 = transformDoubleToMatrix(val3, 1, inputCountPerSample);
//
//    std::cout << "- Prediction des points [0.1;0.1] : " << predict_classification(model, XtoPred1) << std::endl;
//    std::cout << "- Prediction des points [2;2] : " << predict_classification(model, XtoPred2) << std::endl;
//    std::cout << "- Prediction des points [3;3] : " << predict_classification(model, XtoPred3) << std::endl;
//
//    std::cout << std::endl << std::endl;
//    std::cout << ">> Graphe :";
//    std::cout << std::endl << std::endl;
//
//    for (double i = 2; i >= -0.05; i-=0.05) {
//        printf("%4.2f | ", i < 0 ? 0 : i );
//        for (double j = 0; j <= 2.05; j+=0.05) {
//            double XtoPred[2] = {i, j};
//            Matrix<double> m = transformDoubleToMatrix(XtoPred, 1, inputCountPerSample);
//            if(predict_classification(model, m) == 1){
//                std::cout << " - ";
//            }else{
//                std::cout << " x ";
//            }
//        }
//        std::cout << std::endl;
//    }

    /*
     * Regression
     */

    // TODO : Regression

    std::cout <<  "Before regression : " << std::endl;
    std::cout << model << std::endl;
    model = fit_regression(model, XtrainsMatrix, YtrainsMatrix);
    std::cout <<  "After regression : " << std::endl;
    std::cout << model << std::endl;

    double val1[2] = {0.1, 0.1};
    Matrix<double> XtoPred1 = transformDoubleToMatrix(val1, 1, inputCountPerSample);

    double val2[2] = {2, 2};
    Matrix<double> XtoPred2 = transformDoubleToMatrix(val2, 1, inputCountPerSample);

    double val3[2] = {3, 3};
    Matrix<double> XtoPred3 = transformDoubleToMatrix(val3, 1, inputCountPerSample);
//
//    std::cout << "- Prediction des points [0.1;0.1] : " << predict_regression(model, XtoPred1) << std::endl;
//    std::cout << "- Prediction des points [2;2] : " << predict_regression(model, XtoPred2) << std::endl;
//    std::cout << "- Prediction des points [3;3] : " << predict_regression(model, XtoPred3) << std::endl;

}
