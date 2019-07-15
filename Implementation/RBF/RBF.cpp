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
    double gamma;
} RBF;

int getRand(int iter, int max) {
    srand(time(0) * (iter + 1 * 5));
    int size = max;
    int res = rand() % size;
    return res;
}

RBF *initRBF(MatrixXd X, MatrixXd WMatrix, double gamma) {
    RBF *rbf = new RBF();
    rbf->W = WMatrix;
    rbf->X = X;
    rbf->gamma = gamma;
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
    RBF *rbf = initRBF(XMatrix, W, gamma);
    return rbf;
}

SUPEREXPORT double naive_rbf_regression_predict(RBF *rbf, double *sample) {
    MatrixXd sampleMatrix = Map<MatrixXd>(sample, 1, rbf->X.cols());
    MatrixXd gaussianOutputs(1, rbf->X.rows());
    for (int i = 0; i < rbf->X.rows(); i++) {
        gaussianOutputs(0, i) = exp((-rbf->gamma) * (pow(((rbf->X.row(i) - sampleMatrix).norm()), 2)));
    }
    return (gaussianOutputs * rbf->W).sum();
}

SUPEREXPORT int naive_rbf_classification_predict(RBF *rbf, double *sample) {
    if (naive_rbf_regression_predict(rbf, sample) < 0) {
        return -1;
    } else {
        return 1;
    }
}

SUPEREXPORT RBF *rbf_train(
        double *X,
        double *Y,
        int inputCountPerSample,
        int sampleCount,
        int epochs = 5,
        int k = 2,
        double gamma = 100) {

    MatrixXd XMatrix = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(X, inputCountPerSample, sampleCount);
    MatrixXd YMatrix = Map<MatrixXd>(Y, inputCountPerSample, 1);
    MatrixXd KMatrix = MatrixXd::Random(k, sampleCount);
    MatrixXd ClosestMatrix = Map<MatrixXd>(XMatrix.data(), inputCountPerSample, sampleCount);
    RowVectorXd ClosestCenter(sampleCount);
    RowVectorXd AveragePositionCenter(sampleCount);
    RowVectorXd VectorSum;
    double distance;
    double lastDistance;
    int numberOfPointsEqual;

    //Initialize k random centers

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < sampleCount; j++) {
            KMatrix(i, j) = (KMatrix(i, j) + 1) / 2;
        }
    }

    cout << "Kmatrix : " << KMatrix << endl;

    //Repeat for number of epochs
    for (int i = 1; i <= epochs; i++) {
        cout << "epoch " << i << endl;
        lastDistance = -1;
        //For each X row get the closest KMatrix
        for (int j = 0; j < inputCountPerSample; j++) {
            for (int m = 0; m < k; m++) {
                distance = (KMatrix.row(m) - XMatrix.row(j)).norm();

                /*cout << "    Current distance :" << distance << endl;
                cout << "    Current center :" << KMatrix.row(m) << endl;
                cout << "    Current lastDistance :" << lastDistance << endl;*/

                if (lastDistance != -1) {
                    if (distance < lastDistance) {
                        ClosestCenter = KMatrix.row(m);
                    }
                }
                lastDistance = distance;

                ClosestMatrix.row(j) = ClosestCenter;
            }
        }

        //cout << "ClosestMatrix : \n" << ClosestMatrix << endl;

        //Recalculate centers from the individuals
        for (int j = 0; j < k; j++) {
            numberOfPointsEqual = 0;
            VectorSum = RowVectorXd::Zero(sampleCount);
            for (int m = 0; m < inputCountPerSample; m++) {
                if (KMatrix.row(j) == ClosestMatrix.row(m)) {
                    numberOfPointsEqual++;
                    VectorSum += XMatrix.row(m);
                }
            }
            if (numberOfPointsEqual != 0) {
                KMatrix.row(j) = VectorSum / numberOfPointsEqual;
            }
        }
    }

    //Calculate a new phi from XMatrix and KMatrix
    MatrixXd phi(inputCountPerSample, k);
    for (int i = 0; i < inputCountPerSample; i++) {
        for (int j = 0; j < k; j++) {
            phi(i, j) = exp(-gamma * (pow(((XMatrix.row(i) - KMatrix.row(j)).norm()), 2)));
        }
    }


    cout << "Phi transpose : \n" << phi.transpose() << endl;
    cout << "Phi : \n" << phi << endl;
    cout << "Phi transpose * phi : \n" << phi.transpose() * phi << endl;
    cout << "inverse(Phi transpose * phi) : \n" << (phi.transpose() * phi).inverse() << endl;

    //Calculate W
    MatrixXd W = (phi.transpose() * phi).inverse() * phi.transpose() * YMatrix;

    cout << "W : \n" << W << endl;
    //cout << "\n" << W << endl;


    RBF *rbf = initRBF(KMatrix, W, gamma);
    return rbf;
}

SUPEREXPORT double rbf_predict(RBF *rbf, double *sample) {
    MatrixXd sampleMatrix = Map<MatrixXd>(sample, 1, rbf->X.cols());
    MatrixXd gaussianOutputs(1, rbf->X.rows());
    for (int i = 0; i < rbf->X.rows(); i++) {
        gaussianOutputs(0, i) = exp((-rbf->gamma) * (pow(((rbf->X.row(i) - sampleMatrix).norm()), 2)));
    }
    return (gaussianOutputs * rbf->W).sum();
}

SUPEREXPORT double getWSize(RBF *rbf) {
    int res = 0;
    for (int l = 0; l < rbf->W.rows(); l++) {
        res++;
    }
    return res;
}

SUPEREXPORT double *getAllWValues(RBF *rbf) {
    int size = getWSize(rbf);
    auto *res = new double[size];
    for (int i = 0; i < size; i++) {
        res[i] = rbf->W(i, 0);
    }
    return res;
}

SUPEREXPORT RBF *create(double* WValues, int length){
    MatrixXd WMatrix = Map<MatrixXd>(WValues, length, 1);
    return initRBF(WMatrix, WMatrix, 100.0);
}

int main() {
    double gamma = 100;
    int inputCountPerSample = 10;
    int sampleCount = 2;

    double X[200] = {
            0.28326176, 0.69094544,
            0.02930919, 0.555766,
            0.12024438, 0.35618814,
            0.58741242, 0.20598827,
            0.26791684, 0.87068651,
            0.70817648, 0.34697038,
            0.2167688, 0.815847,
            0.64789846, 0.62824209,
            0.07816814, 0.7772395,
            0.71860739, 0.63836079,
            0.57926517, 0.54289996,
            0.07573565, 0.51037785,
            0.26499484, 0.84581008,
            0.42585284, 0.88810611,
            0.85277176, 0.19573003,
            0.15934619, 0.73192273,
            0.16176391, 0.70725274,
            0.01598741, 0.94652365,
            0.93487067, 0.35764317,
            0.87597372, 0.3829717,
            0.90199735, 0.20617172,
            0.02294745, 0.09994164,
            0.74353129, 0.6386675,
            0.53391984, 0.32858542,
            0.94469396, 0.9362246,
            0.4878991, 0.18178962,
            0.21360782, 0.75601664,
            0.03932507, 0.18888541,
            0.69283634, 0.91537383,
            0.76231881, 0.29394792,
            0.05492021, 0.64328716,
            0.59946173, 0.90386575,
            0.16915894, 0.01609477,
            0.78398324, 0.27697256,
            0.97661318, 0.25807922,
            0.37038305, 0.04970378,
            0.45502584, 0.05938631,
            0.52492248, 0.50836815,
            0.10546606, 0.23397258,
            0.2407559, 0.38531204,
            0.64693799, 0.12174754,
            0.29085993, 0.89417024,
            0.55842834, 0.75532969,
            0.19342846, 0.96302601,
            0.06412921, 0.10793609,
            0.07335453, 0.99474073,
            0.89637312, 0.30247975,
            0.89581179, 0.74760349,
            0.77290953, 0.28638938,
            0.30400509, 0.60056855,
            0.56359536, 0.74162032,
            0.2301, 0.17553249,
            0.5204126, 0.20100611,
            0.0392403, 0.80281179,
            0.79087308, 0.73127739,
            0.64837928, 0.06910031,
            0.06709879, 0.40315874,
            0.12751911, 0.17502659,
            0.24367518, 0.72425034,
            0.38447693, 0.86527315,
            0.00937103, 0.83365169,
            0.68921445, 0.04331947,
            0.9417521, 0.39634615,
            0.41388114, 0.96543644,
            0.83730953, 0.21099374,
            0.00172664, 0.63341357,
            0.15948857, 0.71208283,
            0.10375515, 0.96937841,
            0.58090452, 0.16645215,
            0.34724347, 0.34850378,
            0.37941643, 0.14870052,
            0.96830616, 0.1061845,
            0.32200933, 0.50837697,
            0.78506505, 0.06867123,
            0.30304001, 0.539811,
            0.64173924, 0.08572589,
            0.4915084, 0.09707046,
            0.32313497, 0.54427286,
            0.69563607, 0.1101378,
            0.75314428, 0.17715733,
            0.83809649, 0.62516299,
            0.3555431, 0.42132574,
            0.51545058, 0.49509963,
            0.53856322, 0.07173409,
            0.79843056, 0.61691314,
            0.54725031, 0.08880569,
            0.65433769, 0.72883484,
            0.88614601, 0.01408466,
            0.44381219, 0.60281243,
            0.40227897, 0.42534561,
            0.08853207, 0.52134254,
            0.79467357, 0.04678112,
            0.60775271, 0.23242654,
            0.54446513, 0.18258105,
            0.29463552, 0.91775553,
            0.65303336, 0.57510648,
            0.73287643, 0.84377549,
            0.97338964, 0.98474017,
            0.43268874, 0.35449473,
            0.60782024, 0.62358018
    };

    double Y[100] = {
            0.56599192, 0.68220215, 0.25720905, 0.77600784, 0.52871302, 0.35236054,
            0.43632839, 0.32550578, 0.79456349, 0.56027528, 0.12382264, 0.82430519,
            0.47210969, 0.51363324, 0.83089115, 0.17570892, 0.95657055, 0.9767218,
            0.69952698, 0.10705872, 0.76208851, 0.82243979, 0.2492221, 0.93802969,
            0.08418948, 0.57868929, 0.05186721, 0.18925236, 0.24771842, 0.0925256,
            0.5536874, 0.45902611, 0.51971946, 0.52704192, 0.02177463, 0.46034,
            0.10285241, 0.77075103, 0.96600219, 0.82898726, 0.53581565, 0.16737838,
            0.30147697, 0.54130158, 0.36982138, 0.11356511, 0.42775087, 0.7086256,
            0.89255598, 0.19879228, 0.67157162, 0.77975825, 0.89616628, 0.71180037,
            0.71171973, 0.91627995, 0.68064555, 0.69224503, 0.70326362, 0.14895878,
            0.67028804, 0.13142794, 0.49727718, 0.98963302, 0.59928018, 0.21334513,
            0.21878596, 0.14717135, 0.97813619, 0.00773312, 0.37525607, 0.16336265,
            0.81460615, 0.24277097, 0.21507084, 0.73189295, 0.36018627, 0.0239073,
            0.76372235, 0.9319223, 0.14621714, 0.6576305, 0.09925247, 0.96309882,
            0.27511175, 0.28703056, 0.23672181, 0.57924957, 0.04401678, 0.33682471,
            0.54762529, 0.90942542, 0.74479753, 0.30372051, 0.76605893, 0.45648476,
            0.33028456, 0.44437519, 0.85467176, 0.01071847
    };

    double sample[2] = {0.14216546, 0.46456465};


    RBF *rbfModel1 = naive_rbf_train(X, Y, 10, 2, 100);
    double res1 = naive_rbf_regression_predict(rbfModel1, sample);
    cout << "RBF Regression predict : " << res1 << endl;

    int signRes = naive_rbf_classification_predict(rbfModel1, sample);
    cout << "Classification predict : " << signRes << endl;

    cout << "Eigen display" << endl;
    cout << rbfModel1->W << endl;
    cout << "My display" << endl;
    for (int i = 0; i < rbfModel1->W.rows(); i++) {
        cout << getAllWValues(rbfModel1)[i] << endl;
    }
    cout << getWSize(rbfModel1) << endl;

//
//    RBF *rbfModel2 = rbf_train(X, Y, 100, 2, 1000, 20, 100);
//    //double *X, double *Y, int inputCountPerSample, int sampleCount, int epochs = 5, int k = 2,double gamma = 100
//    double res2 = rbf_predict(rbfModel2, sample);
//    cout << res2 << endl;


}

}