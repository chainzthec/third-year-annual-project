import json
import os
import sys

import Implementation.MLP.CLibrary as MLP
import Implementation.Rosenblatt.CLibrary as ROSENBLATT

# import MLP.CLibrary as MLP
# import Rosenblatt.CLibrary as ROSENBLATT
# import RBF.CLibrary as RBF


def load(_filename):
    dirname = os.path.dirname(__file__)
    _filename = os.path.join(dirname, "Models/" + _filename)

    print(_filename)

    with open(_filename, mode='r+') as f:
        content = json.load(f)

    value = "Erreur lors du chargement !"
    algo_name = content['type']
    if algo_name.upper() == "MLP":
        value = MLP.create(content)
    elif algo_name.upper() == "ROSENBLATT":
        value = ROSENBLATT.create(content)

    return value, algo_name


def predict(model, algo_name, XToPredict):
    if algo_name.upper() == "MLP":
        return MLP.predict(model, XToPredict, True)
    elif algo_name.upper() == "ROSENBLATT":
        return ROSENBLATT.predict_classification(model, XToPredict)


def save(model, algo_name, _filename):

    value = {}
    if algo_name.upper() == "MLP":
        value = MLP.export(model)
    elif algo_name.upper() == "ROSENBLATT":
        value = ROSENBLATT.export(model)
    # elif algo_name.upper() == "RBF":
        # value = RBF.export(model)

    json_val = json.dumps(value)

    _fileName = algo_name + "_" + _filename + '.model'
    dirname = os.path.dirname(__file__)
    _fileName = os.path.join(dirname, "Models/" + _fileName)

    file = open(_fileName, "w+")
    file.write(json_val)
    file.close()

    print(_fileName + ' saved')


if __name__ == "__main__":
    X = [
        0, 0,
        1, 0,
        0, 1,
        1, 1,
    ]
    Y = [1, -1, -1, 1]

    # N = [2, 2, 1]
    # sampleCount = 4
    #
    # mlpReg_1 = MLP.init(N)
    # mlpReg_1 = MLP.fit_regression(
    #     mlpReg_1,
    #     X,
    #     Y,
    #     sampleCount,
    #     10000,
    #     0.001
    # )
    #
    # save(mlpReg_1, 'MLP', 'regression_1000_001')

    # fileName = "MLP_regression_1000_001.model"
    # mlpReg_1 = load(fileName)

    # XToPredict = [0, 0]
    # res = MLP.predict(mlpReg_1, XToPredict)
    # print(res)

    # X = [
    #     0, 0,
    #     1, 0,
    #     0, 1,
    #     2, 2,
    #     1, 2,
    #     2, 1,
    #     0.25, 0.25,
    #     0.1, 0.1,
    #     0.15, 0.15,
    #     0.3, 0.3,
    #     3, 3,
    #     1.5, 1.5,
    #     2.5, 2.5
    # ]
    # Y = [-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1]
    #
    # sampleCount = 13
    # inputCountPerSample = 2

    # mlpClas_1 = ROSENBLATT.create_linear_model(inputCountPerSample)
    # mlpClas_1 = ROSENBLATT.fit_classification(
    #     mlpClas_1,
    #     X,
    #     Y,
    #     0.001,
    #     10000
    # )

    # save(mlpClas_1, 'ROSENBLATT', 'classification_1000_001')

    # fileName = "ROSENBLATT_classification_1000_001.model"
    # mlpClas_1 = load(fileName)

    # print(mlpClas_1)
    #
    # XToPredict = [0, 0]
    # res = ROSENBLATT.predict_classification(mlpClas_1, XToPredict)
    # print(res)
    #
    # XToPredict = [1.5, 1.5]
    # res = ROSENBLATT.predict_classification(mlpClas_1, XToPredict)
    # print(res)
