#
# Created by Baptiste Vasseur on 2019-05-18.
#
import os
import sys
from ctypes import *


def get_platform():
    platforms = {
        'linux': 'Linux',
        'darwin': 'OSX',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


dirname = os.path.dirname(__file__)
if get_platform() == "OSX":
    myDll = cdll.LoadLibrary(os.path.join(dirname, 'Librairie/Mac/Rosenblatt_Mac.so'))  # For Mac

elif get_platform() == "Linux":
    myDll = cdll.LoadLibrary(os.path.join(dirname, 'Librairie/Linux/Rosenblatt_Linux.so'))  # For Linux

elif get_platform() == "Windows":
    myDll = cdll.LoadLibrary(os.path.join(dirname, 'Librairie/Windows/Rosenblatt_Windows.so'))  # For Windows

def create_linear_model(inputCountPerSample):
    myDll.create_linear_model.argtypes = [c_int32]
    myDll.create_linear_model.restype = POINTER(ARRAY(c_double, inputCountPerSample + 1))
    createdModel = myDll.create_linear_model(inputCountPerSample)
    return createdModel


def fit_classification(WPointer, XTrain, YTrain, alpha, epochs, inputCountPerSample=False):

    XTrainPointer = (c_double * len(XTrain))(*XTrain)
    YTrainPointer = (c_double * len(YTrain))(*YTrain)

    if not inputCountPerSample:
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

    myDll.fit_classification.restype = POINTER(c_double)
    result = myDll.fit_classification(WPointer, XTrainPointer, YTrainPointer, sampleCount, inputCountPerSample, alpha, epochs)

    values = []
    for i in range(inputCountPerSample + 1):
        values.append(result[i])

    return {"model": values, "inputCountPerSample": inputCountPerSample, "sampleCount": sampleCount,
            'type': 'rosenblatt', 'liner_type': 'classification', "alpha": alpha, 'epochs': epochs}


def predict_classification(W, value):
    inputCountPerSample = len(value)
    valuePointer = (c_double * len(value))(*value)

    WPointer = (c_double * len(W))(*W)

    myDll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, len(W))),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    myDll.predict_classification.restype = c_double
    return myDll.predict_classification(WPointer, valuePointer, inputCountPerSample, True)


def fit_regression(WPointer, XTrain, YTrain, inputCountPerSample=False):
    XTrainPointer = (c_double * len(XTrain))(*XTrain)
    YTrainPointer = (c_double * len(YTrain))(*YTrain)

    if not inputCountPerSample:
        inputCountPerSample = int(len(XTrain) / len(YTrain))

    sampleCount = int(len(XTrain) / inputCountPerSample)

    myDll.fit_regression.argtypes = [
        POINTER(ARRAY(c_double, inputCountPerSample + 1)),
        POINTER(ARRAY(c_double, len(XTrain))),
        POINTER(ARRAY(c_double, len(YTrain))),
        c_int32,
        c_int32
    ]

    myDll.fit_regression.restype = POINTER(c_double)
    result = myDll.fit_regression(WPointer, XTrainPointer, YTrainPointer, sampleCount, inputCountPerSample)

    values = []
    for i in range(inputCountPerSample + 1):
        values.append(result[i])

    return {"model": values, "inputCountPerSample": inputCountPerSample, "sampleCount": sampleCount,
            'type': 'rosenblatt', 'liner_type': 'regression'}


def predict_regression(W, value):
    inputCountPerSample = len(value)
    valuePointer = (c_double * len(value))(*value)

    WPointer = (c_double * len(W))(*W)

    myDll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, len(W))),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    myDll.predict_regression.restype = c_double
    return myDll.predict_regression(WPointer, valuePointer, inputCountPerSample, True)


def export(model):
    return model


def create(content):
    return None