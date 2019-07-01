#
# [27/05/2019] Created by Baptiste Vasseur
# [30/06/2019] Modified by Théo Huchard
#

from ctypes import *
import platform as pf
import json
import os

my_file = open("../settings.json", "r")
path = json.load(my_file)['projet_path'].replace("\\", "/")
my_file.close()
os_name = pf.system()

if os_name == "Darwin":
    myDll = cdll.LoadLibrary(path + "/Implementation/MLP/Librairie/Mac/MultiLayerPerceptron_Mac.so")  # For Mac
elif os_name == "Windows":
    myDll = cdll.LoadLibrary(
        path + "/Implementation/MLP/Librairie/Windows/MultiLayerPerceptron_Windows.dll")  # For Windows
elif os_name == "Linux":
    myDll = cdll.LoadLibrary(path + "/Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so")  # For Linux
else:
    raise ValueError("Error : OS is not supported")


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

    myDll.predict.restype = POINTER(c_double)
    predictions = myDll.predict(mlp, pointr)
    return [predictions[i] for i in range(N[-1])]