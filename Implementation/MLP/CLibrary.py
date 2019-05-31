#
# Created by Baptiste Vasseur on 2019-05-27.
#

from ctypes import *

myDll = cdll.LoadLibrary("./Librairie/Mac/MultiLayerPerceptron_Mac.so")  # For Mac
# myDll = cdll.LoadLibrary("./Librairie/Linux/MultiLayerPerceptron_Linux.so")  # For Linux
# myDll = cdll.LoadLibrary("./Librairie/Windows/MultiLayerPerceptron_Windows.dll")  # For Windows


def init(neurons):
    neuronSize = len(neurons)
    neuronPointer = (c_int32 * neuronSize)(*neurons)

    myDll.init.argtypes = [
        POINTER(ARRAY(c_int32, neuronSize)),
        c_int32
    ]

    myDll.init.restype = c_void_p
    return myDll.init(neuronPointer, neuronSize)


def fit(mlp, XTrain, YTrain, epochs, alpha):

    lenX = len(XTrain)
    lenY = len(YTrain)
    XTrainFinal = (c_double * lenX)(*XTrain)
    YTrainFinal = (c_double * lenY)(*YTrain)

    sampleCounts = int(lenX - lenY)
    inputCountPerSample = int(lenX / lenY)

    myDll.fit.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, lenX)),
        POINTER(ARRAY(c_double, lenY)),
        c_int32,
        c_int32,
        c_int32,
        c_double
    ]

    myDll.fit.restype = c_void_p
    return myDll.fit(mlp, XTrainFinal, YTrainFinal, sampleCounts, inputCountPerSample, epochs, alpha)


def predict(mlp, xToPredict, N):
    inputCountPerSample = len(xToPredict)
    pointr = (c_double * inputCountPerSample)(*xToPredict)

    myDll.predict.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, inputCountPerSample)),
        c_int32,
    ]

    myDll.predict.restype = POINTER(c_double)
    predictions = myDll.predict(mlp, pointr, inputCountPerSample)
    return [predictions[i] for i in range(N[-1])]
