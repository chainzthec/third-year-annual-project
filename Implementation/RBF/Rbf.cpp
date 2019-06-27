#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "../Librairie/Matrix.h"

using std::vector;
using std::cout;
using std::endl;

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
	d = sqrt( pow( (a->x - b->x), 2) + pow( (a->y - b->y), 2) );
	sqrt( pow( (a->x - b->x), 2) + pow( (a->y - b->y), 2) );	
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

	double** teta = new double* [sizeRbf];
	double* tetaToTran = new double[sizeRbf*sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){ 
		teta[i] = new double[sizeRbf];
		
		for(int j = 0;j < sizeRbf; j++ ){
			distancel[i] = distance(&Xtrain[i], &Xtrain[j]);
			teta[i][j] = exp(-myRBF->gamma*pow( distancel[i], 2));
			tetaToTran[i*sizeRbf+j] = teta[i][j];	
			
			
		}
		printf("=/");
	}
	
	Matrix<double>tetaFin = transformDoubleToMatrix(tetaToTran, sizeRbf,sizeRbf);
	Matrix<double>YFin = transformDoubleToMatrix(Ytrain,sizeRbf,1);
	Matrix<double>W = tetaFin.getInverse() * YFin;

	return W.convertToDouble();
}

void regresRBF(point* Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double *res = new double[sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		distancel[i] = distance(&Xtrain[i], &Xtrain[1]);
		
			for(int j = 0;j< sizeRbf; j++){
				res[i] = W[i]* exp((-myRBF->gamma)*pow(distancel[i], 2));		
				
			}
		}
		
	for(int i = 0 ; i< sizeRbf;i++){
		myRBF->Wo[0] +=  res[i];	
	}
	
}

void classifRBF(point* Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double res = 0.0;
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		distancel[i] = distance(&Xtrain[i], &Xtrain[i+1]);
			for(int j = 0;j< sizeRbf; j++){
				res += W[i]* exp((-myRBF->gamma)*pow((distancel[i]), 2));		
			}
		}
	
	myRBF->Wo[0] = sign(res);
}


int main (){
//init XTrain
//init YTrain
	point*XTrain = new point[7];
	
	double YTrain[7][1];
	
	XTrain[0].x = 0.13984698 ;
    XTrain[0].y = 0.41485388;

    XTrain[1].x = 0.28093573 ;
    XTrain[1].y = 0.36177096;

    XTrain[2].x = 0.25704393 ;
    XTrain[2].y = 0.97695092;

    XTrain[3].x = 0.05471647 ;
    XTrain[3].y = 0.8640708;

    XTrain[4].x = 0.91900274;
    XTrain[4].y = 0.95617945;

    XTrain[5].x = 0.1753089;
    XTrain[5].y = 0.67689523;
	
    XTrain[6].x = 0.25784674;
    XTrain[6].y = 0.12366917;
            
  	XTrain[7].x = 0.97495302;
    XTrain[7].y = 0.01277128;
            


    YTrain[0][0] = 0.46119306;
    YTrain[1][0] = 0.78636786;
    YTrain[2][0] = 0.2617359 ;
	YTrain[3][0] = 0.25985246;
	YTrain[4][0] = 0.28554652;
	YTrain[5][0] = 0.57842217;
	YTrain[6][0] = 0.35202585;
	YTrain[7][0] = 0.11248387;


    double W[7]{
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5
    };

   

   RBF* myRBF = initRBF(W, 7, 1);
   double*newWeight = RbfWeight(*YTrain,XTrain,7,myRBF);
   regresRBF(XTrain, newWeight, myRBF,7);
   //classifRBF(XTrain, newWeight, myRBF,4);
   printf("\nreg:%f\n",myRBF->Wo[0] );  
   //printf("classif:%f\n",myRBF->Wo[0] );  


return 0;
}



}