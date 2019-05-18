//
// Created by Théo Huchard on 15/05/19.
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
#include <vector>

using std::vector;
using std::cout;
using std::endl;

double getRand(double min, double max) {
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);
    return val;
}

//Produit entre deux matrices
vector<double> matrixProduct(const vector<double> &matrix1, const vector<double> &matrix2, const int numberOfRowsOfM1,
                             const int numberOfColumnsOfM1, const int numberOfColumnsOfM2) {
    vector<double> result(numberOfRowsOfM1 * numberOfColumnsOfM2);

    //m1 ligne
    for (int i = 0; i < numberOfRowsOfM1; i++) {
        //m2 colonnes
        for (int j = 0; j < numberOfColumnsOfM2; j++) {
            result[i * numberOfColumnsOfM2 + j] = 0.f;
            //m1 colonnes
            for (int k = 0; k < numberOfColumnsOfM1; k++) {
                result[i * numberOfColumnsOfM2 + j] += matrix1[i * numberOfColumnsOfM1 + k] * matrix2[k * numberOfColumnsOfM2 + j];
            }
        }
    }
    return result;
}


// Retourne l'application
vector<double> sigmoid(const vector<double> &matrix) {
    int length = matrix.size();
    vector<double> result(length);
    for (int i = 0; i < length; i++) {
        result[i] = 1 / (1 + exp(-matrix[i]));
    }
    return result;
}

//
vector<double> delta(const vector<double> &matrix1, const vector<double> &matrix2) {
    int length = matrix1.size();
    vector<double> result(length);
    for (int i = 0; i < length; i++) {
        result[i] = matrix1[i] * matrix2[i];
    }
    return result;
}


// Retourne la valeur de la fonction sigmoid dérivée (f'(x) = f(x)(1 - f(x)))
vector<double> sigmoid_derivative(const vector<double> &m1) {
    int length = m1.size();
    vector<double> result(length);
    for (int i = 0; i < length; i++) {
        result[i] = m1[i] * (1 - m1[i]);
    }
    return result;
}

//Retourne la transposée de la matrice
vector<double> transpose(double *matrix, const int numberOfColumns, const int numberOfRows) {
    int i, j;
    vector<double> result(numberOfColumns * numberOfRows);
    //row * col
    for (int x = 0; x < numberOfColumns * numberOfRows; x++) {
        i = x / numberOfColumns;
        j = x % numberOfColumns;
        result[x] = matrix[numberOfRows * j + i];
    }
    return result;
}

// Retourne la somme de deux vecteurs
vector<double> getW(const vector<double> &matrix1, const vector<double> &matrix2) {
    int length = matrix1.size();
    vector<double> result(length);
    for (int i = 0; i < length; ++i) {
        result[i] = matrix1[i] + matrix2[i];
    }
    return result;
}

// Affiche le résultat
void print(const vector<double> &matrix, int numberOfRows, int numberOfColumns) {
    for (int i = 0; i != numberOfRows; ++i) {
        for (int j = 0; j < numberOfColumns; ++j)
            cout << matrix[i * numberOfColumns + j];
        cout << '\n';
    }
    cout << endl;
}

//Récupère l'erreur à partir de YTrain et la prédiction
vector<double> getError(const vector<double> &Y, const vector<double> &prediction) {
    int yVectorSize = Y.size();
    vector<double> predictionError(yVectorSize);

    for (int i = 0; i < yVectorSize; ++i) {
        predictionError[i] = Y[i] - prediction[i];
    }

    return predictionError;
}

void fit_mlp(vector<double> &XTrain, vector<double> &YTrain, vector<double> &W, int epochs){
    cout << "Nombre d'epochs =" << epochs << endl;

    for (int i = 0; i < epochs; i++) {
        // 4 x 4 pour la matrice XTrain et 4 x 1 pour la matrice W (pour que le produit entre XTrain et W soit possible)
        vector<double> prediction = sigmoid(matrixProduct(XTrain, W, 4, 4, 1));
        // récupère l'erreur en faisant une différence entre chaque element de YTrain et chaque element du même indice de prediction
        vector<double> predictionError = getError(YTrain, prediction);
        // récupère le delta en effectuant un produit de chaque element de predictionError et de chaque élement de la dérivée sigmoid de prediction
        vector<double> predictionDelta = delta(predictionError, sigmoid_derivative(prediction));
        // effectue le produit de la transposée par le delta
        vector<double> WDelta = matrixProduct(transpose(&XTrain[0], 4, 4), predictionDelta, 4, 4, 1);
        // effectue la différence entre le poids original et le delta
        W = getW(W, WDelta);
        //affiche le résultat de la prediction sur l'avant dernier élément
        cout << "Epoch " << i + 1 << " :" << endl;
        print(prediction, 4, 1);
    }
}

int main() {

    vector<double> XTrain{
            3.3, 2.2, 1.4, 0.2,
            3.9, 2.0, 1.4, 0.2,
            5.2, 2.4, 4.4, 2.3,
            4.9, 2.0, 4.1, 1.8
    };
    vector<double> YTrain{
            0,
            0,
            1,
            1
    };
    vector<double> W{
            0.5,
            0.5,
            0.5,
            0.5
    };
    int epochs(1000);

    fit_mlp(XTrain, YTrain, W, epochs);

    return 0;
}
