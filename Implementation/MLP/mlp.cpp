#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>

using std::vector;
using std::cout;
using std::endl;


typedef struct MLP_s{
    
    double * input;
    double * hideen;
    double * output;
    double ** Wo;//poids couche sortie
    double ** Wh;//poids couche cachée
    int * npl;//nombre de pers par couche
    double* target;//Y
    double* err;
    
}mlpModel;

static double alpha = 0.5; 


extern "C" {

double getRand(double min, double max) {
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);
    return val;
}

// init la struct mlp
mlpModel* init_mlp(int inputSize, int hiddenSize, int outputSize, double*W){
	
	mlpModel * mlp = new mlpModel();
	mlp->npl = new int[3];
	mlp->npl[0] = inputSize;
	mlp->npl[1] = hiddenSize;
	mlp->npl[2] = outputSize;
	
	
	mlp->target = new double [outputSize];
	mlp->err = new double [outputSize];

	mlp->input = new  double[inputSize];
	mlp->hideen = new  double[hiddenSize];
	mlp->output = new  double[outputSize];
	mlp->Wh = new double*[inputSize+1];
	mlp->Wo = new double*[hiddenSize+1];

	for (int i = 0 ; i < inputSize+1 ; i++){
		mlp->Wh[i] = new double[hiddenSize+1];
	}

	for(int j = 0 ; j < inputSize+1 ; j++){
		for(int k = 0 ; k < hiddenSize+1 ; k++){
			mlp->Wh[j][k] = W[j];//getRand(-1,1);
			mlp->err[k] = 0.0;		
		}
	}

	for (int i = 0 ; i < hiddenSize+1 ; i++){
		mlp->Wo[i] = new double[outputSize+1];
	}


	return mlp;
}
//activation
double sigmoidTanh(double x){//Tanh
	return 1 / (1 + exp(-1 * x));
}

double sigmoidIdt(double x){//identité
	return x;
}



void display (mlpModel *mlp){
	for(int k = 0 ; k < mlp->npl[2] ; k++){
		printf("predict %d: %f\n",k,mlp->output[k] );
		printf("target%d: %f\n",k,mlp->target[k] );
		
		printf("err = target - predict  :%f\n",mlp->err[k] );

	}
}

void ClassifForwardPropagate(mlpModel *mlp , double* XTrain ){

			int bias = 1;
	 
			for(int i = 0 ; i < mlp->npl[0] ; i++){
				mlp->input[i]  = XTrain[i];
			}
			double *Xh = new double[mlp->npl[1]+1];// stocker les x des couches cachées
		// passer les input vers les couches cachées
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				for(int j = 0 ; j < mlp->npl[0] ; j++){
					Xh[i] += mlp->Wh[j][i] * mlp->input[i] + bias;
				}
			}
		// activer les neurones couche cachées
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				mlp->hideen[i] = sigmoidTanh(Xh[i]);
				printf("sig n hid:%f\n",mlp->hideen[i] );
			}
			double *Xo = new double[mlp->npl[2]+1];
		//passer les input vers des couches cachées vers la couche output
			for(int j = 0 ; j < mlp->npl[2] ; j++){
				for(int i = 0 ; i < mlp->npl[1] ; i++){
					Xo[j] += mlp->Wo[j][i] * mlp->hideen[i] + bias;
				}
			}
		// activer les neurones couche sortie
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				mlp->output[k] = sigmoidTanh(Xo[k]);
				printf("sig out:%f\n",mlp->output[k] );
			}
	}

void ClassifBackwardPropagate(mlpModel *mlp ,double* YTrain){
	//calcul l'erreur en sortie
			for(int i = 0; i < mlp->npl[2] ; i++){
				mlp->target[i] = YTrain[i];
				mlp->err[i] = mlp->target[i] - mlp->output[i];
				printf("err:%f\n",mlp->err[i] );
			}
			// calcul gradient couche de sortie
			double ** Wog = new double*[mlp->npl[2]];// poids du gradient de la couche de sortie
			for(int j = 0 ; j< mlp->npl[2] ;j++){
				Wog[j] = new double[mlp->npl[1]];
			}
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					Wog [k][l] = - mlp->err[k] * mlp->output[k] * ( 1 - mlp->output[k] * mlp->hideen[k] ) ;
				}	 
			}			
			//calcul gradient couche cachées
			double ** Whg = new double*[mlp->npl[1]];//poids du gradient des couches cachées
			for(int j = 0 ; j< mlp->npl[1];j++){
				Whg[j] = new double[mlp->npl[0]];
			}
			
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					Whg [k][l] = 0.0;
					double e = 0.0;
					for(int j = 0 ; j< mlp->npl[2];j++){
						e += Whg [k][j] * mlp->err[j];
						Whg[k][l] = -e * mlp->hideen[k] * ( 1 - mlp->hideen[k] * mlp->input[l] ) ; 
					}
				}	
			}
			//mise à jour des poids couche de sortie
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					mlp->Wo [k][l] -= alpha * Wog[k][l];
					
				}
			}
			//mise à jour des poids couche cachée
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					mlp->Wh [k][l] -= alpha * Whg[k][l];
					
				}
			}
			display(mlp);
			
}

void fitCalssification(mlpModel *mlp , double* XTrain ,double* YTrain, int epochs){
	cout << "Nombre d'epochs =" << epochs << endl;
	for(int z = 0 ; z < epochs ; z++){	
			
		ClassifForwardPropagate(mlp,XTrain);
		ClassifBackwardPropagate(mlp,YTrain);

		cout << "Epoch " << z + 1 << " :\n" << endl;
	}		
}

//TODO
void RegresForwardPropagate(mlpModel *mlp , double* XTrain ){

			int bias = 1;
	 
			for(int i = 0 ; i < mlp->npl[0] ; i++){
				mlp->input[i]  = XTrain[i];
			}
			double *Xh = new double[mlp->npl[1]+1];// stocker les x des couches cachées
		// passer les input vers les couches cachées
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				for(int j = 0 ; j < mlp->npl[0] ; j++){
					Xh[i] += mlp->Wh[j][i] * mlp->input[i];
				}
			}
		// activer les neurones couche cachées
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				mlp->hideen[i] = sigmoidTanh(Xh[i]);
				printf("sig n hid:%f\n",mlp->hideen[i] );
			}
			double *Xo = new double[mlp->npl[2]+1];
		//passer les input vers des couches cachées vers la couche output
			for(int j = 0 ; j < mlp->npl[2] ; j++){
				for(int i = 0 ; i < mlp->npl[1] ; i++){
					Xo[j] += mlp->Wo[j][i] * mlp->hideen[i] ;
				}
			}
		// activer les neurones couche sortie
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				mlp->output[k] = sigmoidIdt(Xo[k]);//TODO
				printf("sig out:%f\n",mlp->output[k] );
			}
	}

void RegresfBackwardPropagate(mlpModel *mlp ,double* YTrain){
	//calcul l'erreur en sortie
			for(int i = 0; i < mlp->npl[2] ; i++){
				mlp->target[i] = YTrain[i];
				mlp->err[i] = (mlp->target[i] - mlp->output[i]);//TODO
				printf("err:%f\n",mlp->err[i] );
			}
			// calcul gradient couche de sortie
			double ** Wog = new double*[mlp->npl[2]];// poids du gradient de la couche de sortie
			for(int j = 0 ; j< mlp->npl[2] ;j++){
				Wog[j] = new double[mlp->npl[1]];
			}
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					Wog [k][l] = - mlp->err[k] * mlp->output[k] * ( 1 - mlp->output[k] * mlp->hideen[k] ) ;
				}	 
			}			
			//calcul gradient couche cachées
			double ** Whg = new double*[mlp->npl[1]];//poids du gradient des couches cachées
			for(int j = 0 ; j< mlp->npl[1];j++){
				Whg[j] = new double[mlp->npl[0]];
			}
			
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					Whg [k][l] = 0.0;
					double e = 0.0;
					for(int j = 0 ; j< mlp->npl[2];j++){
						e += Whg [k][j] * mlp->err[j];
						Whg[k][l] = -e * mlp->hideen[k] * ( 1 - mlp->hideen[k] * mlp->input[l] ) ; 
					}
				}	
			}
			//mise à jour des poids couche de sortie
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					mlp->Wo [k][l] -= alpha * Wog[k][l];
					
				}
			}
			//mise à jour des poids couche cachée
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					mlp->Wh [k][l] -= alpha * Whg[k][l];
					
				}
			}
			display(mlp);
			
}

void fitRegression(mlpModel *mlp , double* XTrain ,double* YTrain, int epochs){
	cout << "Nombre d'epochs =" << epochs << endl;
	for(int z = 0 ; z < epochs ; z++){	
			
		RegresForwardPropagate(mlp,XTrain);
		RegresfBackwardPropagate(mlp,YTrain);

		cout << "Epoch " << z + 1 << " :\n" << endl;
	}		
}




int main(){

    double XTrain[4] = {
            3.3, 2.2, 1.4, 0.2
    };
    double YTrain[4] = {
            1,
            0,
            1,
            0
    };
    double W[4]{
            0.5,
            0.5,
            0.5,
            0.5
    };

    int epochs(5000);

    mlpModel *mlp = init_mlp(4,4,4,W);
   	//fitRegression(mlp , XTrain ,YTrain, epochs);
	fitCalssification(mlp , XTrain ,YTrain, epochs);		 
    
    return 0;
}
}









