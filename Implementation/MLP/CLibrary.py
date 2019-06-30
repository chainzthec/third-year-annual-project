#
# [27/05/2019] Created by Baptiste Vasseur
# [30/06/2019] Modified by Théo Huchard
#

from ctypes import *
import platform as pf
import json
import os
from pathlib import Path

path = str(Path().absolute()).replace("\\", "/")
os_name = pf.system()

if os_name == "Darwin":
    my_dll = cdll.LoadLibrary(path + "/MLP/Librairie/Mac/MultiLayerPerceptron_Mac.so")  # For Mac
if os_name == "Linux":
    my_dll = cdll.LoadLibrary(path + "/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so")  # For Linux
if os_name == "Windows":
    my_dll = cdll.LoadLibrary(path + "/MLP/Librairie/Windows/MultiLayerPerceptron_Windows.dll")  # For Windows


def init(neurons):
    neuron_size = len(neurons)
    neuron_pointer = (c_int32 * neuron_size)(*neurons)

    my_dll.init.argtypes = [
        POINTER(ARRAY(c_int32, neuron_size)),
        c_int32
    ]

    my_dll.init.restype = c_void_p
    return my_dll.init(neuron_pointer, neuron_size)


def fit_classification(mlp, x_train, y_train, sample_count, epochs, alpha):
    x_train_final = (c_double * len(x_train))(*x_train)
    y_train_final = (c_double * len(y_train))(*y_train)

    my_dll.fit_classification.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, len(x_train))),
        POINTER(ARRAY(c_double, len(y_train))),
        c_int32,
        c_int32,
        c_double
    ]

    my_dll.fit_classification.restype = c_void_p
    return my_dll.fit_classification(mlp, x_train_final, y_train_final, sample_count, epochs, alpha)


def fit_regression(mlp, x_train, y_train, sample_count, epochs, alpha):
    c_x_train = (c_double * len(x_train))(*x_train)
    c_y_train = (c_double * len(y_train))(*y_train)

    my_dll.fit_regression.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, len(x_train))),
        POINTER(ARRAY(c_double, len(y_train))),
        c_int32,
        c_int32,
        c_double
    ]

    my_dll.fit_regression.restype = c_void_p
    return my_dll.fit_regression(mlp, c_x_train, c_y_train, sample_count, epochs, alpha)


def predict(mlp, x_to_predict, n):
    c_x_to_predict = (c_double * len(x_to_predict))(*x_to_predict)

    my_dll.predict.argtypes = [
        c_void_p,
        POINTER(ARRAY(c_double, len(x_to_predict))),
    ]

    my_dll.predict.restype = POINTER(c_double)
    predictions = my_dll.predict(mlp, c_x_to_predict)
    return [predictions[i] for i in range(n[-1])]