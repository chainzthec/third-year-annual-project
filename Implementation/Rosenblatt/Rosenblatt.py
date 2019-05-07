# from ctypes import *

# def main():
    # dll = CDLL("./libsource.so")
    # dll.create_linear_model.argtypes = [c_int32]
    # dll.create_linear_model.restype = c_void_p
    # linear_model = dll.create_linear_model(2)
    # dll.predict_regression.argtypes = [c_void_p, POINTER(ARRAY(c_double, 2)), c_int32]
    # dll.predict_regression.restype = c_double
    # native_input = (c_double * 2)(*[0.0, 0.0])
    # result = dll.predict_regression(linear_model, native_input, 2)
    # print(linear_model)
    # print(result)

# if __name__ == "__main__":
    # main()
