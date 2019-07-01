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

RBF* create_rbf_model(double *W, int inPutSize, int outPutSize){
	
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

	if(a > 0){return 1;}

	if(a == 0){return 0;}

	if(a < 0){return -1;}
}

double distance(double** a, double** b,int i, int j){
	
	double d = 0;
	d = sqrt( pow( (a[i][0] - b[j][0]), 2) + pow( (a[i][1] - b[i][1]), 2) );
	return d;
}

double getRand(double min, double max) {
    
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);

    return val;
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

double* naive_rbf_train(double *Ytrain, double** Xtrain, int sizeRbf,RBF* myRBF){

	double** phi = new double* [sizeRbf];
	double* phiToTran = new double[sizeRbf*sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){ 
		phi[i] = new double[sizeRbf];
		distancel[i] = distance(Xtrain, Xtrain ,i , i);
		for(int j = 0;j < sizeRbf; j++ ){
			
			phi[i][j] = exp(-myRBF->gamma*pow( distancel[i], 2));
			phiToTran[i*sizeRbf+j] = phi[i][j];	
			
			
		}
		printf("=/");
	}
	
	Matrix<double>phiFin = transformDoubleToMatrix(phiToTran, sizeRbf,sizeRbf);
	Matrix<double>YFin = transformDoubleToMatrix(Ytrain,sizeRbf,1);
	Matrix<double>W = phiFin.getInverse() * YFin;

	return W.convertToDouble();
}

void naive_rbf_predict(double** Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double *res = new double[sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		distancel[i] = distance(Xtrain, Xtrain ,i ,1);
		
			for(int j = 0;j< sizeRbf; j++){
				res[i] = W[i]* exp((-myRBF->gamma)*pow(distancel[i], 2));		
			}
		}
		
	for(int i = 0 ; i< sizeRbf;i++){
		myRBF->Wo[0] +=  res[i];	
	}
	
}

void naive_rbf_classif(double** Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double tmp;
	double *res = new double[sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		distancel[i] = distance(Xtrain, Xtrain ,i ,1);
		for(int j = 0;j< sizeRbf; j++){
			res[i] = W[i]* exp((-myRBF->gamma)*pow(distancel[i], 2));		
				
		}
	}
		
	for(int i = 0 ; i< sizeRbf;i++){
		 tmp += res[i];	
	}

	myRBF->Wo[0] = sign(tmp);
}

double** mu_lloyd_rbf(double** XTrain, int k_mu, int sizeRbf ,double max, double min){
	
	int *i_init = new int [k_mu];
	double sum = 0.0;
	double** mu = new double *[k_mu];
	for(int i = 0; i < k_mu; i ++){
		i_init[i] = getRand(sizeRbf, 0);//TODO :fonction init mu
		mu[i] = new double[k_mu];
		}
		for(int f = 0; f < sizeRbf; f++){
			for(int j = 0; j < k_mu; j++){
				mu[f][j] = XTrain[f][i_init[j]];
			}	
		}
		
	double* S = new double[k_mu];
	for(int k = 0 ; k < k_mu ; k++){
		for(int n = 0 ;n < sizeRbf; n++){
			if( distance(XTrain,mu,n,k) <= distance(XTrain,mu,n,i_init[k]) ){
				i_init[k] = k;	
				S[k] = XTrain[k][n];//avoir
				printf("%f\ni:%d ",S[k],k );
			}
			
			
		}
		
		sum += S[k] * XTrain[k][k];	
	}
	for(int i = 0 ; i < k_mu ; i++){
		for(int k = 0 ; k < k_mu ; k++){

			mu[i][k] = 1/(abs(S[k])) * sum;//TODO :fonction mise à jour mu
		}
	}

	return mu;

}


double* k_mean_rbf_train(double* YTrain, double** XTrain,double** mu, RBF *myRBF, int sizeRbf){

	double** phi = new double* [sizeRbf];
	double* phiToTran = new double[sizeRbf*sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){ 
		phi[i] = new double[sizeRbf];
		distancel[i] = distance(XTrain, mu ,i , i);
		for(int j = 0;j < sizeRbf; j++ ){
			
			phi[i][j] = exp(-myRBF->gamma*pow( distancel[i], 2));
			phiToTran[i*sizeRbf+j] = phi[i][j];	
		}
		printf("=/");
	}
	
	
	Matrix<double>phiFin = transformDoubleToMatrix(phiToTran, sizeRbf,sizeRbf);
	Matrix<double>phiTrans = phiFin.getTranspose();
	Matrix<double>produit = phiTrans * phiFin;
	Matrix<double>YFin = transformDoubleToMatrix(YTrain,sizeRbf,1);
	Matrix<double>W = produit.getInverse() * phiTrans * YFin;

	return W.convertToDouble();

}


void k_mean_rbf_predict(double** Xtrain, double* W, RBF* myRBF, int sizeRbf){
	
	double tmp;
	double *res = new double[sizeRbf];
	double* distancel = new double[sizeRbf];
	for(int i = 0;i < sizeRbf; i++ ){
		distancel[i] = distance(Xtrain, Xtrain ,i ,1);
		
			for(int j = 0;j< sizeRbf; j++){
				res[i] = W[i]* exp((-myRBF->gamma)*pow(distancel[i], 2));		
				
			}
		}
		
	for(int i = 0 ; i< sizeRbf;i++){
		 tmp += res[i];	
	}

	myRBF->Wo[0] = tmp;
	
}


int main (){
//init XTrain
//init YTrain
	double **XTrain = new double*[7];
	for(int i = 0; i< 7 ; i++){
		XTrain[i] = new double [2];
	}
	
	double YTrain[7][1];
	
	XTrain[0][0] = 0.13984698;
    XTrain[0][1] = 0.41485388;

    XTrain[1][0] = 0.28093573;
    XTrain[1][1] = 0.36177096;

    XTrain[2][0] = 0.25704393;
    XTrain[2][1] = 0.97695092;

    XTrain[3][0] = 0.05471647;
    XTrain[3][1] = 0.8640708;

    XTrain[4][0] = 0.91900274;
    XTrain[4][1] = 0.95617945;

    XTrain[5][0] = 0.1753089;
    XTrain[5][1] = 0.67689523;
	
    XTrain[6][0] = 0.25784674;
    XTrain[6][1] = 0.12366917;  

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

   

   	RBF* myRBF = create_rbf_model(W, 7, 1);
   	//double **mu = mu_lloyd_rbf(XTrain,7 ,3 ,0.05, 0.97 );
   	double*model =  naive_rbf_train (*YTrain,XTrain,7,myRBF);//k_mean_rbf_train(*YTrain, XTrain, mu, myRBF,7);
   	naive_rbf_predict(XTrain, model, myRBF,7);

   	//k_mean_rbf_train(*YTrain, XTrain,mu, myRBF,7);
   	//k_mean_rbf_predict(XTrain, model, myRBF,7);
   	//naive_rbf_classifRBF(XTrain, newWeight, myRBF,4);
   	printf("\nreg:%f\n",myRBF->Wo[0] );  
   	//printf("classif:%f\n",myRBF->Wo[0] );  
	free(myRBF);

return 0;
}



}