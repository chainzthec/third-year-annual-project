#
# Created by Baptiste Vasseur on 2019-05-27.
#
import CLibrary as CLib


def main():
    epochs = 50000
    alpha = 0.001

    N = [2, 2, 1]

    X = [
        0, 0,
        1, 0,
        0, 1,
        1, 1
    ]

    Y = [1, -1, -1, 1]

    inputCountPerSample = int(len(X) / len(Y))
    sampleCount = int(len(Y))

    mlp = CLib.init(N)

    # mlp = CLib.fit(mlp, XTrainFinal, YTrainFinal, sampleCount, inputCountPerSample, epochs, alpha)

    # predictions = CLib.predict(mlp, [0, 0])


if __name__ == "__main__":
    main()
