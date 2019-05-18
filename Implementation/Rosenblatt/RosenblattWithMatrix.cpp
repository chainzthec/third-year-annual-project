////
//// Created by Baptiste Vasseur on 2019-05-01.
////
//#if _WIN32
//#define SUPEREXPORT __declspec(dllexport)
//#else
//#define SUPEREXPORT
//#endif
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <cmath>
//#include <iostream>
//
//#include "../Librairie/Matrix.h"
//
////extern "C" {
//
///*
// *
// * Utils
// *
// */
//
//// ajouter une autre fonction sign(somme de 0 à sampleCount) (w_i * x_i) ** 0
//// bias = position de la droite, a et b orientation de la droite
//
//SUPEREXPORT double getRand(double min, double max) {
//    double val = (double) rand() / RAND_MAX;
//    val = min + val * (max - min);
//
////    std::cout << "Rand : " << val << std::endl;
//    return val;
//}
//
//SUPEREXPORT Matrix<double> create_linear_model(int inputCountPerSample) {
//    Matrix<double> matrix(1, inputCountPerSample+1);
//
//    for (int i = 0; i < inputCountPerSample+1; ++i) {
//        double r = getRand(-1.0, 1.0);
//        matrix.put(0, i, r);
//    }
//
//    return matrix;
//}
//
//SUPEREXPORT Matrix<double> addBias(const Matrix<double> &matrix) {
//
//    int bias = 1;
//    Matrix<double> result(matrix.getRows(), matrix.getColumns() + 1);
//
//    for (int i = 0; i < matrix.getRows(); ++i) {
//        result.put(i, 0, bias);
//        for (int j = 0; j < matrix.getColumns(); ++j) {
//            result.put(i, j + 1, matrix.get(i, j));
//        }
//    }
//
//    return result;
//}
//
///*
// *
// * Implementation
// *
// */
//
//SUPEREXPORT double predict_regression(
//        const Matrix<double>& W, // TODO : need to be inline
//        const Matrix<double>& XLineToPredict
//) {
////    std::cout << std::endl << "W:" << std::endl << W << std::endl << "XLineToPredict: " << std::endl << XLineToPredict << std::endl;
//
//    // TODO : Inférence (CF Slides !)
//
//    double res = 0.0;
//    for (int j = 0; j < XLineToPredict.getColumns(); ++j) {
//        res = res + (W.get(0, j) * XLineToPredict.get(0, j));
//    }
//
//    return res >= 0 ? 1.0 : -1.0 ;
//}
//
//SUPEREXPORT double predict_classification(
//        const Matrix<double> &W,
//        const Matrix<double> &XLineToPredict
//) {
//    return predict_regression(W, XLineToPredict);
//}
//
//SUPEREXPORT Matrix<double> fit_regression(
//        const Matrix<double> &W,
//        const Matrix<double> &XTrain,
//        const Matrix<double> &YTrain
//) {
//    // TODO : entrainement (correction des W, cf slides !)
//
//    Matrix<double> XTrainFin = addBias(XTrain);
////    std::cout << "Xtrain :" << std::endl << XTrainFin << std::endl;
//
//    Matrix<double> XTrainTranspose = XTrainFin.getTranspose();
////    std::cout  << "XtrainTranspose :" << std::endl << XTrainTranspose << std::endl;
//
//    Matrix<double> XTmultX = XTrainTranspose * XTrainFin;
////    std::cout << "XTmultX :" << std::endl << XTmultX << std::endl;
//
//    Matrix<double> XTmultXInverse = XTmultX.getInverse();
////    std::cout << "XTmultXInverse :" << std::endl << XTmultXInverse << std::endl;
//
////    std::cout << "Return :" << std::endl << XTmultXInverse * XTrainTranspose * YTrain << std::endl;
//    return (XTmultXInverse * XTrainTranspose * YTrain).getTranspose();
//}
//
//
//SUPEREXPORT Matrix<double> fit_classification(
//        Matrix<double> W,
//        const Matrix<double> &XTrain,
//        const Matrix<double> &YTrain,
//        double alpha, // Learning Rate (0,01 par exemple)
//        int epochs // Nombre d'itération
//) {
//    Matrix<double> XTrainFin = addBias(XTrain);
//
//    int sampleCount = XTrain.getRows(); // Nombres images
//
//    for (int i = 0; i < epochs; i++) {
//        for (int j = 0; j < sampleCount; j++) {
//
//            //TODO : entrainement (correction des W, cf slides !)
//            Matrix<double> tmp = XTrainFin.getRow(j);
//
//            for (int k = 0; k < XTrainFin.getColumns(); k++) {
//                double predict = predict_classification(W, tmp);
//                double value = W.get(0, k) + alpha * (YTrain.get(j, 0) - predict) * XTrainFin.get(j, k);
//                W.put(0, k, value);
//            }
//        }
//    }
//
//    return W;
//}
//
//SUPEREXPORT void delete_linear_model(const double *W) {
//    delete[] W;
//}
//
//SUPEREXPORT Matrix<double> transformDoubleToMatrix(double *mat, int rows, int cols = 1) {
//    int currentPos = 0;
//    Matrix<double> res(rows, cols);
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            res.put(i, j, mat[currentPos]);
//            currentPos += 1;
//        }
//    }
//
//    return res;
//}
//
//SUPEREXPORT Matrix<double> init(double* value, int inputCountPerSample){
//    Matrix<double> matrix = transformDoubleToMatrix(value, 1, inputCountPerSample);
//    matrix = addBias(matrix);
//    return matrix;
//}
//
//
//
//int main() {
//
//    // TODO : main & test
//
//    srand(time(nullptr)); // Enable rand() function
//
//    int inputCountPerSample = 2;
//    int sampleCount = 13;
//    int epochs = 5000;
//    double alpha = 0.001;
//
//    Matrix<double> model = create_linear_model(inputCountPerSample);
//
////    std::cout << "model: " << std::endl << model << std::endl;
//
//    // 26 car sampleCount * inputCountPerSample = 13 * 2 = 26 (soit 13 images de 2 pixels donc 26 pixels au total)
//    double Xtrains[26] = {
//            0, 0,
//            1, 0,
//            0, 1,
//            2, 2,
//            1, 2,
//            2, 1,
//            0.25, 0.25,
//            0.1, 0.1,
//            0.15, 0.15,
//            0.3, 0.3,
//            3, 3,
//            1.5, 1.5,
//            2.5, 2.5
//    };
//
//    // 13 car sampleCount = 13 (soit 13 images)
//    double Ytrains[13] = {-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1};
//
//    Matrix<double> XtrainsMatrix = transformDoubleToMatrix(Xtrains, sampleCount, inputCountPerSample);
//    Matrix<double> YtrainsMatrix = transformDoubleToMatrix(Ytrains, sampleCount, 1);
//
//    /*
//     * Classification
//     */
//
//    std::cout << std::endl <<  "Classification : " << std::endl;
//
//    std::cout <<  "Before Rosenblatt : " << std::endl;
//    std::cout << model << std::endl;
//    Matrix<double> modelClas = fit_classification(model, XtrainsMatrix, YtrainsMatrix, alpha, epochs);
//    std::cout <<  "After Rosenblatt : " << std::endl;
//    std::cout << modelClas << std::endl;
//
//    double val1[2] = {0.25, 0.25};
//    Matrix<double> XtoPred1 = init(val1, inputCountPerSample);
//
//    double val2[2] = {2.5, 2.5};
//    Matrix<double> XtoPred2 = init(val2, inputCountPerSample);
//
//    double val3[2] = {1, 2};
//    Matrix<double> XtoPred3 = init(val3, inputCountPerSample);
//
//    std::cout << "- Prediction des points [0.25;0.25] (-1) : " << predict_classification(modelClas, XtoPred1) << std::endl; // -1
//    std::cout << "- Prediction des points [2.5;2.5] (1) : " << predict_classification(modelClas, XtoPred2) << std::endl; // 1
//    std::cout << "- Prediction des points [1;2] : (1) " << predict_classification(modelClas, XtoPred3) << std::endl; // 1
//
//    /*
//     * Regression
//     */
//
//    std::cout << std::endl <<  "Regression : " << std::endl;
//
//    std::cout << "Before regression : " << std::endl;
//    std::cout << model << std::endl;
//    Matrix<double> modelReg = fit_regression(model, XtrainsMatrix, YtrainsMatrix);
//    std::cout << "After regression : " << std::endl;
//    std::cout << modelReg << std::endl;
//
//    double val4[2] = {0, 0};
//    Matrix<double> XtoPred4 = init(val4, inputCountPerSample);
//
//    double val5[2] = {2.5, 2.5};
//    Matrix<double> XtoPred5 = init(val5, inputCountPerSample);
//
//    double val6[2] = {1.5, 1.5};
//    Matrix<double> XtoPred6 = init(val6, inputCountPerSample);
//
//    std::cout << "- Régression des points [0;0] (-1) : " << predict_regression(modelReg, XtoPred4) << std::endl; // -1
//    std::cout << "- Régression des points [2.5;2.5] (1) : " << predict_regression(modelReg, XtoPred5) << std::endl; // 1
//    std::cout << "- Régression des points [1.5;1.5] (1) : " << predict_regression(modelReg, XtoPred6) << std::endl; // 1
//
//    /*
//     * Graphe
//     */
//
//    // Classification
//
////    std::cout << std::endl << std::endl;
////    std::cout << ">> Graphe Classification :";
////    std::cout << std::endl << std::endl;
////
////    for (double i = 2; i >= -0.05; i-=0.05) {
////        printf("%4.2f | ", i < 0 ? 0 : i );
////        for (double j = 0; j <= 2.05; j+=0.05) {
////            double value[2] = {i, j};
////            Matrix<double> xToPred = init(value, inputCountPerSample);
////            if(predict_classification(model, xToPred) == 1){
////                std::cout << " - ";
////            }else{
////                std::cout << " x ";
////            }
////        }
////        std::cout << std::endl;
////    }
//
//    // Regression
//
////    std::cout << std::endl << std::endl;
////    std::cout << ">> Graphe Regression :";
////    std::cout << std::endl << std::endl;
////
////    for (double i = 2; i >= -0.05; i-=0.05) {
////        printf("%4.2f | ", i < 0 ? 0 : i );
////        for (double j = 0; j <= 2.05; j+=0.05) {
////            double value[2] = {i, j};
////            Matrix<double> xToPred = init(value, inputCountPerSample);
////            if(predict_regression(model, xToPred) == 1){
////                std::cout << " - ";
////            }else{
////                std::cout << " x ";
////            }
////        }
////        std::cout << std::endl;
////    }
//
//
//}
//
////}