#
# Created by Baptiste Vasseur on 2019-05-27.
#

from ctypes import *

myDll = cdll.LoadLibrary("./MultiLayerPerceptron.so")


def init_XTrain(XTrain, sampleCount, inputCountPerSample):
    XTrainPointer = convertToMatrix(XTrain, sampleCount, inputCountPerSample)
    #
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


def init_mlp(neurons):
    neuronPointer = (c_double * len(neurons))(*neurons)
    print(neuronPointer)
    myDll.init_mlp.argtypes = [
        POINTER(ARRAY(c_double, len(neurons))),
        c_int32
    ]

    myDll.init_mlp.restype = c_void_p
    mlp = myDll.init_mlp(neuronPointer, 3)

    # return mlp

    # myDll.init_model.argtypes = [
    #     c_void_p,
    #     c_int32,
    #     c_int32
    # ]
    #
    # myDll.init_model.restype = c_void_p
    # return myDll.init_model(mlp)


