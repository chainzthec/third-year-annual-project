#
# Created by Baptiste Vasseur on 2019-05-27.
#

from ctypes import *

myDll = cdll.LoadLibrary("./MultiLayerPerceptron.so")


def init_XTrain(XTrain, sampleCount, inputCountPerSample):
    XTrainPointer = convertToMatrix(XTrain, sampleCount, inputCountPerSample)

    myDll.addMatrixBias.argtypes = [
        c_void_p,
        c_int32,
        c_int32
    ]
    myDll.addMatrixBias.restype = c_void_p
    return myDll.addMatrixBias(XTrainPointer, sampleCount, inputCountPerSample)


def init_YTrain(YTrain, sampleCount):
    return convertToMatrix(YTrain, sampleCount, 1)


def convertToMatrix(tab, sampleCount, inputCountPerSample):
    tabPointer = (c_double * len(tab))(*tab)
    myDll.convertToMatrix.argtypes = [
        POINTER(ARRAY(c_double, len(tab))),
        c_int32,
        c_int32
    ]

    myDll.convertToMatrix.restype = c_void_p
    return myDll.convertToMatrix(tabPointer, sampleCount, inputCountPerSample)


def fit(neurons, XTrainFinal, YTrainFinal, sampleCounts, epochs, alpha):

    neuronSize = len(neurons)
    neuronPointer = (c_double * neuronSize)(*neurons)

    myDll.fit.argtypes = [
        POINTER(ARRAY(c_double, len(neurons))),
        c_int32,
        c_void_p,
        c_void_p,
        c_int32,
        c_int32,
        c_double
    ]

    myDll.fit.restype = c_void_p
    return myDll.fit(neuronPointer, neuronSize, XTrainFinal, YTrainFinal, sampleCounts, epochs, alpha)


def predict(mlp, xToPredict):
    inputCountPerSample = len(xToPredict)
    pointr = (c_double * inputCountPerSample)(*xToPredict)

    myDll.predict.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, inputCountPerSample)),
        c_int32,
    ]

    myDll.predict.restype = c_void_p
    return myDll.predict(mlp, pointr, inputCountPerSample)


