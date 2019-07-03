#
# [30/06/2019] Created by Théo Huchard
#

from ctypes import *
import platform as pf
import json
import os

my_file = open("../settings.json","r")
path = json.load(my_file)['projet_path'].replace("\\","/")
my_file.close()
os_name = pf.system()

if os_name == "Darwin":
    my_dll = cdll.LoadLibrary(path + "/Implementation/RBF/Librairie/Mac/RBF_Mac.so")  # For Mac
elif os_name == "Windows":
    my_dll = cdll.LoadLibrary(path + "/Implementation/RBF/Librairie/Windows/RBF_Windows.dll")  # For Windows
elif os_name == "Linux":
    my_dll = cdll.LoadLibrary(path + "/Implementation/RBF/Librairie/Linux/RBF_Linux.so")  # For Linux
else:
    raise ValueError("Error : OS is not supported")


def rbf_predict(model, sample):
    my_dll.rbf_predict.argtypes = [
        c_void_p,
        POINTER(c_double)
    ]
    my_dll.rbf_predict.restype = POINTER(c_double)
    predict_value = my_dll.rbf_predict(model, sample)
    return predict_value


def rbf_train(x, y, input_count_per_sample, sample_count, epochs=100, k=2, use_bias=False):
    my_dll.rbf_train.argtypes = [
        POINTER(c_double, len(x)),
        POINTER(c_double, len(y)),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_double),
        POINTER(c_bool)
    ]
    my_dll.rbf_predict.restype = c_void_p
    model = my_dll.rbf_train(x, y, input_count_per_sample, sample_count, epochs, k, use_bias)
    return model


def naive_rbf_train(x, y, input_count_per_sample, sample_count, gamma=100):
    c_x_double = (c_double * len(x))(*x)
    c_y_double = (c_double * len(y))(*y)

    my_dll.naive_rbf_train.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int32,
        c_int32,
        c_double
    ]
    my_dll.naive_rbf_train.restype = c_void_p
    model = my_dll.naive_rbf_train(
        c_x_double,
        c_y_double,
        input_count_per_sample,
        sample_count,
        gamma,
    )
    return model


def naive_rbf_predict(model, sample):
    c_sample_double = (c_double * len(sample))(*sample)

    my_dll.naive_rbf_predict.argtypes = [
        c_void_p,
        POINTER(c_double)
    ]
    my_dll.naive_rbf_predict.restype = c_int32
    predict_value = my_dll.naive_rbf_predict(model, c_sample_double)
    return predict_value
