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
	int sizeRbfOut;
	//double lloydCenter;
	double gamma;

}RBF;

RBF* initRBF(double *W, int inPutSize, int outPutSize){
	RBF* myRBF = new RBF();
	myRBF->Wi = new double[inPutSize];
	myRBF->Wo = new double[outPutSize];
	myRBF->sizeRbfOut = outPutSize;
	myRBF->gamma = 100;

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

double distance(point* a, point* b){
	
	double d = 0;
	d += sqrt(pow((a->x - b->x), 2) + pow((a->y - b->y), 2) );	
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

double* RbfWeight(double *Ytrain, point* Xtrain, int sizeRbf,RBF* myRBF){

	double** phi = new double* [sizeRbf];
	double* phiToTran = new double[sizeRbf*sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){

		phi[i] = new double[sizeRbf];
		
		for(int j = 0;j < sizeRbf; j++ ){
		
			phi[i][j] = exp(-myRBF->gamma*pow( distance(&Xtrain[i],&Xtrain[j]), 2));
			phiToTran[i*sizeRbf+j] = phi[i][j];	
			printf("la%f\n" ,distance(&Xtrain[i],&Xtrain[j]) );

		}
		
	}
	
	Matrix<double>phiFin = transformDoubleToMatrix(phiToTran, sizeRbf,sizeRbf);
	Matrix<double>YFin = transformDoubleToMatrix(Ytrain,sizeRbf,1);
	Matrix<double>W = phiFin.getInverse() * YFin;

	return W.convertToDouble();
}

void regresRBF(point* Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double res;
		for(int i = 0;i< sizeRbf; i++){
			for(int j = 0;j< sizeRbf; j++){
			
				res += W[i]* exp(-myRBF->gamma*pow((distance(&Xtrain[i],&Xtrain[1])), 2));
				printf("%f\n",res );
			}
		}
		
	myRBF->Wo[0] =  res;
}

void classifRBF(point* Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double res = 0.0;
		for(int i = 0;i< sizeRbf; i++){
			for(int j = 0;j< sizeRbf; j++){
				res += W[i]* exp((-myRBF->gamma)*pow((distance(&Xtrain[i],&Xtrain[1])), 2));		
			}
		}
	
	myRBF->Wo[0] = sign(res);
}

void print(RBF *myRBF){
	for(int i = 0; i < myRBF->sizeRbfOut; i++){
		if(myRBF->Wo[i] == 1){
			printf("*");		
		
	}else{
		printf("/");
	}
}

}
double getRand(double min, double max) {
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);

    return val;
}

int main (){
//init XTrain
//init YTrain
	point* XTrain = new point[4];
	
	double YTrain[4][1];
	
	XTrain[0].x = 0.13984698 ;
    XTrain[0].y = 0.41485388;

    XTrain[1].x = 0.28093573 ;
    XTrain[1].y = 0.36177096;

    XTrain[2].x = 0.25704393 ;
    XTrain[2].y = 0.97695092;

    XTrain[3].x = 0.05471647 ;
    XTrain[3].y = 0.8640708;

    YTrain[0][0] = 0.46119306;
 
 
 
    YTrain[1][0] = 0.78636786;
    YTrain[2][0] = 0.2617359 ;
    YTrain[3][0] = 0.25985246;

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
   print(myRBF);


return 0;
}
}
/*
}

[[0.13984698 0.41485388]
[0.28093573 0.36177096]
[0.25704393 0.97695092]
[0.05471647 0.8640708 ]
[0.91900274 0.95617945]
[0.1753089 0.67689523]
[0.25784674 0.12366917]
[0.97495302 0.01277128]
[0.08287882 0.94833339]
[0.39418121 0.79789368]]

[
 0.46119306
 0.78636786
 0.2617359 
 0.25985246
 0.28554652
 0.57842217
 0.35202585
 0.11248387
 0.72196561
 0.60782134
 ]
*/