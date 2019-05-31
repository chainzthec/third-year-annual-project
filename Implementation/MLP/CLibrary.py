#
# Created by Baptiste Vasseur on 2019-05-27.
#

from ctypes import *

myDll = cdll.LoadLibrary("./Librairie/Mac/MultiLayerPerceptron_Mac.so")  # For Mac
# myDll = cdll.LoadLibrary("./Librairie/Linux/MultiLayerPerceptron_Linux.so")  # For Linux
# myDll = cdll.LoadLibrary("./Librairie/Windows/MultiLayerPerceptron_Windows.dll")  # For Windows


def init(neurons):
    neuronSize = len(neurons)
    neuronPointer = (c_double * neuronSize)(*neurons)

    myDll.init.argtypes = [
        POINTER(ARRAY(c_double, neuronSize)),
        c_int32
    ]

    myDll.init.restype = c_void_p
    return myDll.init(neuronPointer, neuronSize)


# def fit(neurons, XTrainFinal, YTrainFinal, sampleCounts, epochs, alpha):
#
#     neuronSize = len(neurons)
#     neuronPointer = (c_double * neuronSize)(*neurons)
#
#     myDll.fit.argtypes = [
#         POINTER(ARRAY(c_double, neuronSize)),
#         c_int32,
#         c_void_p,
#         c_void_p,
#         c_int32,
#         c_int32,
#         c_double
#     ]
#
#     myDll.fit.restype = c_void_p
#     return myDll.fit(neuronPointer, neuronSize, XTrainFinal, YTrainFinal, sampleCounts, epochs, alpha)
#
#
# def predict(mlp, xToPredict):
#     inputCountPerSample = len(xToPredict)
#     pointr = (c_double * inputCountPerSample)(*xToPredict)
#
#     myDll.predict.argtypes = [
#         c_void_p,
#         POINTER(ARRAY(c_double, inputCountPerSample)),
#         c_int32,
#     ]
#
#     myDll.predict.restype = c_void_p
#     return myDll.predict(mlp, pointr, inputCountPerSample)
