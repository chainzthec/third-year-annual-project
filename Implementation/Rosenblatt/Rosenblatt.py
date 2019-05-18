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

    model = CLib.create_linear_model(inputCountPerSample)

    print("Before Rosenblatt : ")
    CLib.displayMatrix(model, 1, inputCountPerSample + 1)
    CLib.fit_classification(model, X, Y, 0.001, 5000)
    print("After Rosenblatt : ")
    CLib.displayMatrix(model, 1, inputCountPerSample + 1)

    val1 = [0.25, 0.25]
    res = CLib.predict_classification(model, val1)
    print("- Prediction des points [0.25;0.25] (-1) : " + str(res))

    val2 = [2.5, 2.5]
    res = CLib.predict_classification(model, val2)
    print("- Prediction des points [2.5;2.5] (1) : " + str(res))

    val3 = [1, 2]
    res = CLib.predict_classification(model, val3)
    print("- Prediction des points [1;2] : (1) " + str(res))


if __name__ == "__main__":
    main()
