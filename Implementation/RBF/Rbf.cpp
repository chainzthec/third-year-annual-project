#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "../Librairie/Matrix.h"

using std::vector;
using std::cout;
using std::endl;

//reg : out = la somme des poids exp(-target |Ypre-pre(n)|²)
//classif: out = sign (reg) 
// la correction des poids W = l'inverse(omega)*Y
//omega : matrice des exp(-y *|Xln-Xn|)

extern "C" {

typedef struct point_s{
	double x;
	double y;
}point;

typedef struct sampleRBF{
	
	double* Wo;
	double* Wi;
	//double lloydCenter;
	double gamma;

}RBF;

RBF* initRBF(double *W, int inPutSize, int outPutSize){
	RBF* myRBF = new RBF();
	myRBF->Wi = new double[inPutSize];
	myRBF->Wo = new double[outPutSize];
	myRBF->gamma = 0.2;

	for(int i = 0;i < inPutSize; i++){
		
			myRBF->Wi[i] = W[i];
	}

	return myRBF;
}


double sign(double a){
	if(a > 0){
		return 1;
	}
	if(a == 0){
		return 0;
	}
	if(a < 0){
		return -1;
	}
}

double distance(double** a, double **b,int size){
	double d = 0;
	for(int i = 0 ; i< size-1; i++){
		for(int j = 0 ; j< 1; j++){
			//printf("a[%d][%d]:%f;b[%d+1][%d+1]:%f\n",a[i][j],b[i+1][j],i,j,i+1,j+1 );
			d += sqrt(pow((a[i][j] - b[i+1][j]), 2) + pow((a[i][j+1] - b[i+1][j+1]), 2) );	
		}
	}
	
	return d;
}

Matrix<double> transformDoubleToMatrix(double *mat, int rows, int cols = 1) {
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

double* RbfWeight(double *Ytrain, double** Xtrain, int sizeRbf,RBF* myRBF){

	double** teta = new double* [sizeRbf];
	double* tetaToTran = new double[sizeRbf*sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		teta[i] = new double[sizeRbf];
		
		for(int j = 0;j < sizeRbf; j++ ){
		
			teta[i][j] = exp((-myRBF->gamma)*pow( distance(Xtrain,Xtrain,sizeRbf), 2));
			tetaToTran[i*sizeRbf+j] = teta[i][j];	
				
		}
		printf("%f\n",distance(Xtrain,Xtrain,sizeRbf) );
	}
	
	Matrix<double>tetaFin = transformDoubleToMatrix(tetaToTran, sizeRbf,sizeRbf);
	Matrix<double>YFin = transformDoubleToMatrix(Ytrain,sizeRbf,1);
	Matrix<double>W = tetaFin.getInverse() * YFin;

	return W.convertToDouble();
}

void regresRBF(double** Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double res = 0.0;
		for(int i = 0;i< sizeRbf; i++){
			for(int j = 0;j< sizeRbf; j++){
				res += W[i]* exp((-myRBF->gamma)*pow((distance(Xtrain,Xtrain,sizeRbf)), 2));		
			}
		}
		
	myRBF->Wo[0] =  res;
}

void classifRBF(double** Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double res = 0.0;
		for(int i = 0;i< sizeRbf; i++){
			for(int j = 0;j< sizeRbf; j++){
				res += W[i]* exp((-myRBF->gamma)*pow((distance(Xtrain,Xtrain,sizeRbf)), 2));		
			}
		}
	
	myRBF->Wo[0] = sign(res);
}


int main (){
//init XTrain
//init YTrain
	double **XTrain = new double*[4];
	for(int i = 0; i< 4;i++){
		XTrain[i] = new double[2];
	}
	double YTrain[4][1];
	
	XTrain[0][0] = 0;
    XTrain[0][1] = 0;

    XTrain[1][0] = 1;
    XTrain[1][1] = 1;

    XTrain[2][0] = 1;
    XTrain[2][1] = 2;

    XTrain[3][0] = 3;
    XTrain[3][1] = 0;

    YTrain[0][0] = 1;
    YTrain[1][0] = -1;
    YTrain[2][0] = 1;
    YTrain[3][0] = 1;

    double W[4]{
            0.5,
            0.5,
            0.5,
            0.5
    };

   

   RBF* myRBF = initRBF(W, 4, 1);
   double*newWeight = RbfWeight(*YTrain,XTrain,4,myRBF);
   regresRBF(XTrain, newWeight, myRBF,4);
   //classifRBF(XTrain, newWeight, myRBF,4);
   printf("reg:%f\n",myRBF->Wo[0] );  
   //printf("classif:%f\n",myRBF->Wo[0] );  


return 0;
}



}