#
# [18/05/2019] Created by Baptiste Vasseur
# [30/06/2019] Modified by Théo Huchard
#

from ctypes import *
import platform as pf
from pathlib import Path

path = str(Path().absolute()).replace("\\", "/")

os_name = pf.system()

if os_name == "Darwin":
    my_dll = cdll.LoadLibrary(path + "/Rosenblatt/Librairie/Mac/Rosenblatt_Mac.so")  # For Mac
elif os_name == "Windows":
    my_dll = cdll.LoadLibrary(path + "/Rosenblatt/Librairie/Windows/Rosenblatt_Windows.dll")  # For windows
elif os_name == "Linux":
    my_dll = cdll.LoadLibrary(path + "/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so")  # For Linux
else:
    raise ValueError("Error : OS is not supported")


def create_linear_model(input_count_per_sample):
    my_dll.create_linear_model.argtypes = [c_int32]
    my_dll.create_linear_model.restype = POINTER(ARRAY(c_double, input_count_per_sample + 1))
    model = my_dll.create_linear_model(input_count_per_sample)
    return model


def fit_classification(w_pointer, x_train, y_train, alpha, epochs):
    x_train_pointer = (c_double * len(x_train))(*x_train)
    y_train_pointer = (c_double * len(y_train))(*y_train)

    input_count_per_sample = int(len(x_train) / len(y_train))
    sample_count = int(len(x_train) / input_count_per_sample)

    my_dll.fit_classification.argtypes = [
        POINTER(ARRAY(c_double, input_count_per_sample + 1)),
        POINTER(ARRAY(c_double, len(x_train))),
        POINTER(ARRAY(c_double, len(y_train))),
        c_int32,
        c_int32,
        c_double,
        c_int32
    ]

    my_dll.fit_classification.restype = POINTER(ARRAY(c_double, input_count_per_sample + 1))
    return my_dll.fit_classification(w_pointer, x_train_pointer, y_train_pointer, sample_count, input_count_per_sample,
                                     alpha, epochs)


def predict_classification(w_pointer, value):
    input_count_per_sample = len(value)
    value_pointer = (c_double * len(value))(*value)

    my_dll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, input_count_per_sample + 1)),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    my_dll.predict_classification.restype = c_double
    return my_dll.predict_classification(w_pointer, value_pointer, input_count_per_sample, True)


def fit_regression(w_pointer, x_train, y_train):
    x_train_pointer = (c_double * len(x_train))(*x_train)
    y_train_pointer = (c_double * len(y_train))(*y_train)

    input_count_per_sample = int(len(x_train) / len(y_train))
    sample_count = int(len(x_train) / input_count_per_sample)

    my_dll.fit_regression.argtypes = [
        POINTER(ARRAY(c_double, input_count_per_sample + 1)),
        POINTER(ARRAY(c_double, len(x_train))),
        POINTER(ARRAY(c_double, len(y_train))),
        c_int32,
        c_int32
    ]

    my_dll.fit_regression.restype = POINTER(ARRAY(c_double, input_count_per_sample + 1))
    return my_dll.fit_regression(w_pointer, x_train_pointer, y_train_pointer, sample_count, input_count_per_sample)


def predict_regression(w_pointer, value):
    input_count_per_sample = len(value)
    value_pointer = (c_double * len(value))(*value)

    my_dll.predict_regression.argtypes = [
        POINTER(ARRAY(c_double, input_count_per_sample + 1)),
        POINTER(ARRAY(c_double, len(value))),
        c_int32,
        c_bool
    ]

    my_dll.predict_regression.restype = c_double
    return my_dll.predict_regression(w_pointer, value_pointer, input_count_per_sample, True)


def delete_linear_model(w, length):
    w_pointer = (c_double * length)(*w)

    my_dll.delete_linear_model.argtype = POINTER(ARRAY(c_double, length))
    my_dll.delete_linear_model.restype = c_void_p
    my_dll.delete_linear_model(w_pointer)


def display_matrix(pointer_val, rows, cols):
    my_dll.displayMatrix.argtypes = [
        POINTER(ARRAY(c_double, cols * rows)),
        c_int32,
        c_int32
    ]

    my_dll.displayMatrix.restype = c_void_p
    my_dll.displayMatrix(pointer_val, rows, cols)


def launch_classification_text(model, value, expected):
    res = predict_classification(model, value)
    print("- Prediction des points " + str(value) + "  :  (" + str(expected) + ") -> (" + str(int(res)) + ")")


def launch_regression_text(model, value, expected):
    res = predict_regression(model, value)
    print("- Prediction des points " + str(value) + "  :  (" + str(expected) + ") -> (" + str(int(res)) + ")")
