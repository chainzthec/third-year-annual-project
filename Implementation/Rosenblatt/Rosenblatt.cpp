//
// Created by Baptiste Vasseur on 2019-05-01.
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

/*
 *
 * Utils
 *
 */

// ajouter une autre fonction sign(somme de 0 à sampleCount) (w_i * x_i) ** 0
// bias = position de la droite, a et b orientation de la droite

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

SUPEREXPORT Matrix<double> transformDoubleToMatrix(double *mat, int rows, int cols = 1) {
    int currentPos = 0;
    Matrix<double> res(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res.put(i, j, mat[currentPos]);
            currentPos += 1;
        }
    }

    return res;
}

SUPEREXPORT void displayMatrix(double* matrix, int rows, int cols){
    int count = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[count] << " - ";
            count++;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

SUPEREXPORT double* addBias(double* mat, int lines, int cols) {
    int bias = 1;

    Matrix<double> matrix = transformDoubleToMatrix(mat, lines, cols);
    Matrix<double> result(matrix.getRows(), matrix.getColumns() + 1);

    for (int i = 0; i < matrix.getRows(); ++i) {
        result.put(i, 0, bias);
        for (int j = 0; j < matrix.getColumns(); ++j) {
            result.put(i, j + 1, matrix.get(i, j));
        }
    }

    double* res = result.convertToDouble();
    return res;
}

/*
 *
 * Implementation
 *
 */

SUPEREXPORT int sign(double val){
    return (val >= 0.0) ? 1 : -1 ;
}

SUPEREXPORT double predict(
        const double* W,
        double* XLineToPredict,
        int inputCountPerSample,
        bool predictState = true
) {
    if(predictState){
        XLineToPredict = addBias(XLineToPredict, 1, inputCountPerSample);
        inputCountPerSample += 1;
    }

    double res = 0.0;
    for (int j = 0; j < inputCountPerSample; ++j) {
        res = res + (W[j] * XLineToPredict[j]);
    }

    return sign(res) ;
}

SUPEREXPORT double predict_regression(
        double* W,
        double* XLineToPredict,
        int inputCountPerSample,
        bool predictState = true
) {
    return predict(W, XLineToPredict, inputCountPerSample, predictState);
}

SUPEREXPORT double predict_classification(
        double* W,
        double* XLineToPredict,
        int inputCountPerSample,
        bool predictState = true
) {
    return predict(W, XLineToPredict, inputCountPerSample, predictState);
}

SUPEREXPORT double* fit_regression(
        double *W,
        double *XTrain,
        double *YTrain,
        int sampleCount, // nombre d'image (ligne)
        int inputCountPerSample //nombre de pixel par img (colonne)
) {
    XTrain = addBias(XTrain, sampleCount, inputCountPerSample);

    Matrix<double> XTrainFin = transformDoubleToMatrix(XTrain, sampleCount, inputCountPerSample);
    Matrix<double> YTrainFin = transformDoubleToMatrix(YTrain, sampleCount, 1);
//    std::cout << "XTrainFin :" << std::endl << XTrainFin << std::endl;
//    std::cout << "YTrainFin :" << std::endl << YTrainFin << std::endl;

    Matrix<double> XTrainTranspose = XTrainFin.getTranspose();
//    std::cout  << "XtrainTranspose :" << std::endl << XTrainTranspose << std::endl;

    Matrix<double> XTmultX = XTrainTranspose * XTrainFin;
//    std::cout << "XTmultX :" << std::endl << XTmultX << std::endl;

    Matrix<double> XTmultXInverse = XTmultX.getInverse();
//    std::cout << "XTmultXInverse :" << std::endl << XTmultXInverse << std::endl;

//    std::cout << "Return :" << std::endl << XTmultXInverse * XTrainTranspose * YTrain << std::endl;

    Matrix<double> res = (XTmultXInverse * XTrainTranspose * YTrainFin).getTranspose();
    return res.convertToDouble();
}

SUPEREXPORT double** oneDimToTwo(const double* XTrain, int ligne, int col){
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


SUPEREXPORT double* fit_classification(
        double* W,
        double* XTrain,
        const double* YTrain,
        int sampleCount, // nombre d'image (ligne)
        int inputCountPerSample, //nombre de pixel par img (colonne)
        double alpha, // Learning Rate (0,01 par exemple)
        int epochs // Nombre d'itération
) {
    XTrain = addBias(XTrain, sampleCount, inputCountPerSample);
    inputCountPerSample += 1;
    double** XTrainFin = oneDimToTwo(XTrain, sampleCount, inputCountPerSample);

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < sampleCount; j++) {
            auto tmp = XTrainFin[j];

            for (int k = 0; k < inputCountPerSample; k++) {
                double predict = predict_classification(W, tmp, inputCountPerSample, false);
                auto value = W[k] + alpha * (YTrain[j] - predict) * XTrainFin[j][k];
                W[k] = value;
            }
        }
    }

    return W;
}

SUPEREXPORT void delete_linear_model(const double *W) {
    delete[] W;
}

int main() {

    srand(time(nullptr)); // Enable rand() function

    int inputCountPerSample = 2;
    int sampleCount = 13;
    int epochs = 5000;
    double alpha = 0.001;

    double* model = create_linear_model(inputCountPerSample);

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


    /*
     * Classification
     */

    std::cout << std::endl <<  "Classification : " << std::endl;

    std::cout <<  "Before Rosenblatt : " << std::endl;
    displayMatrix(model, 1, inputCountPerSample+1);
    double* modelClas = fit_classification(model, Xtrains, Ytrains, sampleCount, inputCountPerSample, alpha, epochs);
    std::cout <<  "After Rosenblatt : " << std::endl;
    displayMatrix(modelClas, 1, inputCountPerSample+1);

    double val1[2] = {0.25, 0.25};
    double val2[2] = {2.5, 2.5};
    double val3[2] = {1, 2};

    std::cout << "- Prediction des points [0.25;0.25] (-1) : " << predict_classification(modelClas, val1, inputCountPerSample) << std::endl; // -1
    std::cout << "- Prediction des points [2.5;2.5] (1) : " << predict_classification(modelClas, val2, inputCountPerSample) << std::endl; // 1
    std::cout << "- Prediction des points [1;2] : (1) " << predict_classification(modelClas, val3, inputCountPerSample) << std::endl; // 1

    /*
     * Regression
     */

    std::cout << std::endl <<  "Regression : " << std::endl;

    std::cout << "Before regression : " << std::endl;
    displayMatrix(model, 1, inputCountPerSample+1);
    double* modelReg = fit_regression(model, Xtrains, Ytrains, sampleCount, inputCountPerSample);
    std::cout << "After regression : " << std::endl;
    displayMatrix(modelReg, 1, inputCountPerSample+1);

    double val4[2] = {0, 0};
    double val5[2] = {2.5, 2.5};
    double val6[2] = {1.5, 1.5};

    std::cout << "- Régression des points [0;0] (-1) : " << predict_regression(modelReg, val4, inputCountPerSample) << std::endl; // -1
    std::cout << "- Régression des points [2.5;2.5] (1) : " << predict_regression(modelReg, val5, inputCountPerSample) << std::endl; // 1
    std::cout << "- Régression des points [1.5;1.5] (1) : " << predict_regression(modelReg, val6, inputCountPerSample) << std::endl; // 1

    /*
     * Graphe
     */

    // Classification

//    std::cout << std::endl << std::endl;
//    std::cout << ">> Graphe Classification :";
//    std::cout << std::endl << std::endl;
//
//    for (double i = 2; i >= -0.05; i-=0.05) {
//        printf("%4.2f | ", i < 0 ? 0 : i );
//        for (double j = 0; j <= 2.05; j+=0.05) {
//            double value[2] = {i, j};
//            if(predict_classification(model, value, 2) == 1){
//                std::cout << " - ";
//            }else{
//                std::cout << " x ";
//            }
//        }
//        std::cout << std::endl;
//    }

    // Regression

//    std::cout << std::endl << std::endl;
//    std::cout << ">> Graphe Regression :";
//    std::cout << std::endl << std::endl;
//
//    for (double i = 2; i >= -0.05; i-=0.05) {
//        printf("%4.2f | ", i < 0 ? 0 : i );
//        for (double j = 0; j <= 2.05; j+=0.05) {
//            double value[2] = {i, j};
//            if(predict_regression(model, value, 2) == 1){
//                std::cout << " - ";
//            }else{
//                std::cout << " x ";
//            }
//        }
//        std::cout << std::endl;
//    }


}

//}