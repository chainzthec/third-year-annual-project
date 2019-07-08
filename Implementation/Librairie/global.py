import json
import os
import sys

import MLP.CLibrary as MLP
import Rosenblatt.CLibrary as ROSENBLATT
# import RBF.CLibrary as RBF


def load(_fileName):
    dirname = os.path.dirname(__file__)
    _fileName = os.path.join(dirname, "../../Models/" + _fileName)

    with open(_fileName, mode='r+') as f:
        content = json.load(f)

    value = "Erreur lors du chargement !"
    typeAlgo = content['type']
    if typeAlgo.upper() == "MLP":
        value = MLP.create(content)
    # elif typeAlgo.upper() == "ROSENBLATT":
    #     value = ROSENBLATT.create(content)

    return value


def save(model, typeAlgo, _fileName):

    value = {}
    if typeAlgo.upper() == "MLP":
        value = MLP.export(model)
    elif typeAlgo.upper() == "ROSENBLATT":
        value = ROSENBLATT.export(model)
    # elif typeAlgo.upper() == "RBF":
        # value = RBF.export(model)

    jsonVal = json.dumps(value)

    _fileName = typeAlgo + "_" + _fileName + '.model'
    dirname = os.path.dirname(__file__)
    _fileName = os.path.join(dirname, "../../Models/" + _fileName)

    file = open(_fileName, "w+")
    file.write(jsonVal)
    file.close()

    print(_fileName + ' saved')


if __name__ == "__main__":
    # X = [
    #     0, 0,
    #     1, 0,
    #     0, 1,
    #     1, 1,
    # ]
    # Y = [1, -1, -1, 1]
    #
    # N = [2, 2, 1]
    # sampleCount = 4
    #
    # mlpReg_1 = MLP.init(N)
    # MLP.fit_regression(
    #     mlpReg_1,
    #     X,
    #     Y,
    #     sampleCount,
    #     10000,
    #     0.001
    # )
    #
    # save(mlpReg_1, 'MLP', 'regression_1000_001')

    fileName = "MLP_regression_1000_001.model"
    mlpReg_1 = load(fileName)

    XToPredict = [0, 0]
    res = MLP.predict(mlpReg_1, XToPredict)
    print(res)

