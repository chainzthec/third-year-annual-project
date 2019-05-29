#
# Created by Baptiste Vasseur on 2019-05-27.
#
import CLibrary as CLib


def main():

    X = [
        0, 0,
        1, 0,
        0, 1,
        1, 1
    ]

    Y = [1, -1, -1, 1]

    N = [2, 2, 1]

    inputCountPerSample = int(len(X) / len(Y))
    sampleCount = int(len(Y))

    XTrainFinal = CLib.init_XTrain(X, sampleCount, inputCountPerSample)
    YTrainFinal = CLib.init_YTrain(Y, sampleCount)

    mlp = CLib.init_mlp(N)


if __name__ == "__main__":
    main()
