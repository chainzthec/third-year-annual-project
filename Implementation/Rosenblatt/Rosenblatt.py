#
# Created by Baptiste Vasseur on 2019-05-01.
#

from ctypes import *


def main():
    dll = cdll.LoadLibrary("./Rosenblatt.so")

    xtrain = (c_double * 26)(
        *[0, 0, 1, 0, 0, 1, 2, 2, 1, 2, 2, 1, 0.25, 0.25, 0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 3, 3, 1.5, 1.5, 2.5, 2.5])
    ytrain = (c_double * 13)(*[-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1])

    inputtest = (c_double * 2)(*[0.0, 0.0])

    samplecount = 13
    inputcountpersample = 2
    alpha = 0.001
    epochs = 5000

    dll.create_linear_model.argtypes = [c_int32]
    dll.create_linear_model.restype = c_void_p
    linear_model = dll.create_linear_model(2)

    dll.fit_classification.argtypes = [c_void_p, POINTER(ARRAY(c_double, 26)), POINTER(ARRAY(c_double, 13)), c_int32,
                                       c_int32, c_float, c_int32]
    dll.fit_classification.restype = c_double

    modelclass = dll.fit_classification(linear_model, xtrain, ytrain, samplecount, inputcountpersample, alpha, epochs)

    # dll.predict_regression.argtypes = [c_void_p, POINTER(ARRAY(c_double, 2)), c_int32]
    # dll.predict_regression.restype = c_double
    # result = dll.predict_regression(modelclass, inputtest, 2)

    # print(modelclass)
    # print(result)


if __name__ == "__main__":
    main()
