#
# Created by Baptiste Vasseur on 2019-05-18.
#

from ctypes import *

myDll = cdll.LoadLibrary("./Rosenblatt.so")


def create_linear_model(inputCountPerSample):
    myDll.create_linear_model.argtypes = [c_int32]
    myDll.create_linear_model.restype = POINTER(ARRAY(c_double, inputCountPerSample + 1))
    createdModel = myDll.create_linear_model(inputCountPerSample)
    return createdModel


def fit_classification(WPointer, XTrain, YTrain, alpha, epochs):
    XTrainPointer = (c_double * len(XTrain))(*XTrain)
    YTrainPointer = (c_double * len(YTrain))(*YTrain)

    inputCountPerSample = int(len(XTrain) / len(YTrain))
    sampleCount = int(len(XTrain) / inputCountPerSample)

    myDll.fit_classification.argtypes = [
        POINTER(ARRAY(c_double, inputCountPerSample + 1)),
        POINTER(ARRAY(c_double, len(XTrain))),
        POINTER(ARRAY(c_double, len(YTrain))),
        c_int32,
        c_int32,
        c_double,
        c_int32
    ]

    myDll.fit_classification.restype = POINTER(ARRAY(c_double, inputCountPerSample + 1))
    return myDll.fit_classification(WPointer, XTrainPointer, YTrainPointer, sampleCount, inputCountPerSample, alpha, epochs)


def predict_classification(WPointer, value):
    inputCountPerSample = len(value)
    valuePointer = (c_double * len(value))(*value)

    myDll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, inputCountPerSample + 1)),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    myDll.predict_classification.restype = c_double
    return myDll.predict_classification(WPointer, valuePointer, inputCountPerSample, True)


def fit_regression(WPointer, XTrain, YTrain):
    XTrainPointer = (c_double * len(XTrain))(*XTrain)
    YTrainPointer = (c_double * len(YTrain))(*YTrain)

    inputCountPerSample = int(len(XTrain) / len(YTrain))
    sampleCount = int(len(XTrain) / inputCountPerSample)

    myDll.fit_regression.argtypes = [
        POINTER(ARRAY(c_double, inputCountPerSample + 1)),
        POINTER(ARRAY(c_double, len(XTrain))),
        POINTER(ARRAY(c_double, len(YTrain))),
        c_int32,
        c_int32
    ]

    myDll.fit_regression.restype = POINTER(ARRAY(c_double, inputCountPerSample + 1))
    return myDll.fit_regression(WPointer, XTrainPointer, YTrainPointer, sampleCount, inputCountPerSample)


def predict_regression(WPointer, value):
    inputCountPerSample = len(value)
    valuePointer = (c_double * len(value))(*value)

    myDll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, inputCountPerSample + 1)),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    myDll.predict_regression.restype = c_double
    return myDll.predict_regression(WPointer, valuePointer, inputCountPerSample, True)


def delete_linear_model(W, length):
    WPointer = (c_double * length)(*W)

    myDll.delete_linear_model.argtype = POINTER(ARRAY(c_double, length))
    myDll.delete_linear_model.restype = c_void_p
    myDll.delete_linear_model(WPointer)


def displayMatrix(pointerVal, rows, cols):

    myDll.displayMatrix.argtypes = [
        POINTER(ARRAY(c_double, cols * rows)),
        c_int32,
        c_int32
    ]

    myDll.displayMatrix.restype = c_void_p
    myDll.displayMatrix(pointerVal, rows, cols)
