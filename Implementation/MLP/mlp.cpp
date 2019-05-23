#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>

using std::cout;
using std::endl;


typedef struct MLP_s{
    
    double * input;// tab x d'entrées
    double * hideen;//tab x des cachées
    double * output;//tab x de sorties
    double ** W;
    int * npl;
    int layer_count;
    double* target;
    double* err;
    
}mlpModel;

static double alpha = 0.5; 


extern "C" {

double getRand(double min, double max) {
    double val = (double) rand() / RAND_MAX;
    val = min + val * (max - min);
    return val;
}


mlpModel* init_mlp(int inputSize, int hiddenSize, int outputSize,int count){
	mlpModel * mlp = new mlpModel();
	mlp->npl = new int[3];
	mlp->npl[0] = inputSize;
	mlp->npl[1] = hiddenSize;
	mlp->npl[2] = outputSize;
	mlp->layer_count = count;
	
	mlp->target = new double [outputSize];
	mlp->err = new double [outputSize];

	mlp->input = new  double[inputSize];
	mlp->hideen = new  double[hiddenSize];
	mlp->output = new  double[outputSize];
	mlp->W = new double*[inputSize+1];
	
	for (int i = 0 ; i < inputSize+1 ; i++){
		mlp->W[i] = new double[hiddenSize+1];
	}

	for(int j = 0 ; j < inputSize+1 ; j++){
		for(int k = 0 ; k < outputSize+1 ; k++){
			mlp->W[j][k] = getRand(-1,1);
			mlp->err[k] = 0.0;		
		}
	}

	return mlp;
}

double sigmoid(double x){
	return 1 / (1 + exp(-1 * x));
}


void display (mlpModel *mlp){
	for(int k = 0 ; k < mlp->npl[2] ; k++){
		printf("out %d: %f\n",k,mlp->output[k] );
		printf("err = target - output :%f\n",mlp->target[k] - mlp->output[k] );

	}
}

void propagate(mlpModel *mlp , double* XTrain ){

	
			for(int i = 0 ; i < mlp->npl[0] ; i++){
				mlp->input[i]  = XTrain[i];
			}
			double *Xh = new double[mlp->npl[1]+1];
		// passer les input vers les couches cachées
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				for(int j = 0 ; j < mlp->npl[0] ; j++){
					Xh[i] += mlp->W[j][i] * XTrain[i];
				}
			}
		// activer les neurones
			for(int i = 0 ; i < mlp->npl[1] ; i++){
				mlp->hideen[i] = sigmoid(Xh[i]);
			}
			double *X = new double[mlp->npl[2]+1];
		//passer les input vers des couches cachées vers la couche output
			for(int j = 0 ; j < mlp->npl[2] ; j++){
				for(int i = 0 ; i < mlp->npl[1] ; i++){
					Xh[i] += mlp->W[j][i] * mlp->hideen[i];
				}
			}
		// activer les neurones
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				mlp->output[k] = sigmoid(Xh[k]);
			}
		 
		 display(mlp);
		
	}

double* learn(mlpModel *mlp ,double*YTrain,int epochs ){

		//calcul l'erreur en sortie
			for(int i = 0; i < mlp->npl[2] ; i++){
				mlp->target[i] = YTrain[i];
				mlp->err[i] = mlp->target[i] - mlp->output[i];
			}
			// calcul gradient couche de sortie
			double ** Wog = new double*[mlp->npl[2]];
			for(int j = 0 ; j< mlp->npl[2] ;j++){
				Wog[j] = new double[mlp->npl[1]];
			}
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					Wog [k][l] = - mlp->err[k] * mlp->output[k];
				}	
			}			
			//calcul gradient couche de cachée
			double ** Whg = new double*[mlp->npl[1]];
			for(int j = 0 ; j< mlp->npl[1];j++){
				Whg[j] = new double[mlp->npl[0]];
			}
			
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					Whg [k][l] = 0.0;
					double e = 0.0;
					for(int j = 0 ; j< mlp->npl[2];j++){
						e += Whg [k][j] * mlp->err[j];
						Whg[k][l] = -e * mlp->hideen[k] * (1 - mlp->hideen[k] * mlp->input[l]); 
						
					}
				}	
			}
			//mise à jour des poids couche de sortie
			for(int k = 0 ; k < mlp->npl[2] ; k++){
				for(int l = 0 ; l < mlp->npl[1] ; l++){
					mlp->W [k][l] -= alpha * Wog[k][l];
					printf("couche sortie:%f\n\n",mlp->W [k][l] );
				}
			}
			//mise à jour des poids couche cachée
			for(int k = 0 ; k < mlp->npl[1] ; k++){
				for(int l = 0 ; l < mlp->npl[0] ; l++){
					mlp->W [k][l] -= alpha * Whg[k][l];
					printf("couche cachée:%f\n\n",mlp->W [k][l] );
				}
			}
	
			}
		}
	


int main(){

    double XTrain[16] = {
            3.3, 2.2, 1.4, 0.2,
            3.9, 2.0, 1.4, 0.2,
            5.2, 2.4, 4.4, 2.3,
            4.9, 2.0, 4.1, 1.8
    };
    double YTrain[2] = {
            1,
            0
    };
    double W[4]{
            0.5,
            0.5,
            0.5,
            0.5
    };

    int epochs(10);

    mlpModel *mlp = init_mlp(4,4,2,0);
    for(int l = 0 ; l < epochs ; l++){
		cout << "Nombre d'epochs =" << epochs << endl;
    	propagate(mlp, XTrain);
    	learn(mlp,YTrain, epochs);
    	cout << "Epoch " << l + 1 << " :" << endl;
    }
    //display(mlp);
    return 0;
}









