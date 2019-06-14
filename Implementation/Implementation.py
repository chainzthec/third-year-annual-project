from ctypes import *
import numpy as np
import platform as pf
import subprocess
	
def fit(
	x_train=None,
	y_train=None,
	sample_count=None,
	epochs=10,
	alpha=None,
	gamma=None,
	method="LINEAR"
):
	osName = pf.system()
	print("OS détécté : ",osName)
	if osName == "Darwin":
		subprocess.call([r'C:\Users\thuchard\Documents\00 - Tools\projet-annuel\Tools\generateDLLMac.sh'])
		# TODO ajouter la ligne pour utiliser generateDLLMac.sh
		mlpDll = cdll.LoadLibrary("./MLP/Librairie/Mac/MultiLayerPerceptron_Mac.so")
	elif osName == "Windows":
		subprocess.call([r'C:\Users\thuchard\Documents\00 - Tools\projet-annuel\Tools\generateDLLWindows.sh'])
		# TODO ajouter la ligne pour utiliser generateDLLWindows.bat
		mlpDll = cdll.LoadLibrary("./MLP/Librairie/Windows/MultiLayerPerceptron_Windows.dll")
	elif osName == "Linux":
		subprocess.call([r'C:\Users\thuchard\Documents\00 - Tools\projet-annuel\Tools\generateDLLLinux.sh'])
		# TODO ajouter la ligne pour utiliser generateDLLLinux.sh
		mlpDll = cdll.LoadLibrary("./MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so")
	else:
		raise ValueError("Error : OS is not supported")

	if method == "MLP":
		len_x = len(x_train)
		len_y = len(y_train)
		x_train_final = (c_double * len_x)(*x_train)
		y_train_final = (c_double * len_y)(*y_train)
		if mlpDll != None:
			neuron_size = len(neurons)
			neuron_pointer = (c_int32 * neuron_size)(*neurons)
			mlpDll.init.argtypes = [POINTER(ARRAY(c_int32,neuron_size)),c_int32]
			mlpDll.fit_classification.argtypes = [
				c_void_p,
				POINTER(ARRAY(c_double, lenX)),
				POINTER(ARRAY(c_double, lenY)),
				c_int32,
				c_int32,
				c_double
			]
			mlpDll.fit_classification.restype = c_void_p
			return mlpDll.fit_classification(mlp, XTrainFinal, YTrainFinal, sampleCount, epochs, alpha)
		else:
			raise ValueError("Error : Couldn't load .so and/or .dll file")

def init(neurons):
    neuronSize = len(neurons)
    neuronPointer = (c_int32 * neuronSize)(*neurons)

    myDll.init.argtypes = [
        POINTER(ARRAY(c_int32, neuronSize)),
        c_int32
    ]

    myDll.init.restype = c_void_p
    return myDll.init(neuronPointer, neuronSize)


def fit_classification(mlp, XTrain, YTrain, sampleCount, epochs, alpha):

    lenX = len(XTrain)
    lenY = len(YTrain)
    XTrainFinal = (c_double * lenX)(*XTrain)
    YTrainFinal = (c_double * lenY)(*YTrain)

    myDll.fit_classification.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, lenX)),
        POINTER(ARRAY(c_double, lenY)),
        c_int32,
        c_int32,
        c_double
    ]

    myDll.fit_classification.restype = c_void_p
    return myDll.fit_classification(mlp, XTrainFinal, YTrainFinal, sampleCount, epochs, alpha)


def fit_regression(mlp, XTrain, YTrain, sampleCount, epochs, alpha):

    lenX = len(XTrain)
    lenY = len(YTrain)
    XTrainFinal = (c_double * lenX)(*XTrain)
    YTrainFinal = (c_double * lenY)(*YTrain)

    myDll.fit_regression.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, lenX)),
        POINTER(ARRAY(c_double, lenY)),
        c_int32,
        c_int32,
        c_double
    ]

    myDll.fit_regression.restype = c_void_p
    return myDll.fit_regression(mlp, XTrainFinal, YTrainFinal, sampleCount, epochs, alpha)


def predict(mlp, xToPredict, N):
    pointr = (c_double * len(xToPredict))(*xToPredict)

    myDll.predict.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, len(xToPredict))),
    ]

    myDll.predict.restype = POINTER(c_int32)
    predictions = myDll.predict(mlp, pointr)
    return [predictions[i] for i in range(N[-1])]


x_train = [
        1,2,
        0.5,0.4
]

y_train = [
        0.4,0.6,
        0.7,0.8
]

sample_count = 4
epochs = 10
alpha = None
gamma = 0.1
method = "MLP"

fit(x_train,y_train,sample_count,epochs,gamma=None,method=method)