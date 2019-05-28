#
# Created by Baptiste Vasseur on 2019-05-01.
#
import CLibrary as CLib


def main():

    X = [
        0, 0,
        1, 2,
        1, 0,
        0, 1,
        2, 2,
        2, 1,
        0.25, 0.25,
        0.1, 0.1,
        0.15, 0.15,
        0.3, 0.3,
        3, 3,
        1.5, 1.5,
        2.5, 2.5
    ]

    Y = [-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1]

    inputCountPerSample = int(len(X) / len(Y))

    #
    #
    #

    modelClassif = CLib.create_linear_model(inputCountPerSample)

    print("\nBefore Classification : ")
    CLib.displayMatrix(modelClassif, 1, inputCountPerSample + 1)
    trainedModelClassif = CLib.fit_classification(modelClassif, X, Y, 0.001, 5000)
    print("After Classification : ")
    CLib.displayMatrix(trainedModelClassif, 1, inputCountPerSample + 1)

    launchClassificationText(trainedModelClassif, [0.25, 0.25], -1)
    launchClassificationText(trainedModelClassif, [2.5, 2.5], 1)
    launchClassificationText(trainedModelClassif, [1, 2], 1)
    launchClassificationText(trainedModelClassif, [0.3, 0.3], -1)
    launchClassificationText(trainedModelClassif, [3, 3], 1)

    #
    #
    #

    modelReg = CLib.create_linear_model(inputCountPerSample)

    print("\nBefore Regression : ")
    CLib.displayMatrix(modelReg, 1, inputCountPerSample + 1)
    trainedModelReg = CLib.fit_regression(modelReg, X, Y)
    print("After Regression : ")
    CLib.displayMatrix(trainedModelReg, 1, inputCountPerSample + 1)

    launchRegressionText(trainedModelReg, [0, 0], -1)
    launchRegressionText(trainedModelReg, [2.5, 2.5], 1)
    launchRegressionText(trainedModelReg, [1.5, 1.5], 1)
    launchRegressionText(trainedModelReg, [0.3, 0.3], -1)
    launchRegressionText(trainedModelReg, [3, 3], 1)


def launchClassificationText(model, value, expected):
    res = CLib.predict_classification(model, value)
    print("- Prediction des points "+str(value)+"  :  ("+str(expected)+") -> (" + str(int(res)) + ")")


def launchRegressionText(model, value, expected):
    res = CLib.predict_regression(model, value)
    print("- Prediction des points "+str(value)+"  :  ("+str(expected)+") -> (" + str(int(res)) + ")")


if __name__ == "__main__":
    main()
