#
# Created by Baptiste Vasseur on 2019-05-27.
#
import CLibrary as CLib


def main():
    epochs = 50000
    alpha = 0.001

    # N = [2, 2, 1]
    #
    # X = [
    #     0, 0,
    #     1, 0,
    #     0, 1,
    #     1, 1
    # ]
    #
    # Y = [1, -1, -1, 1]
    #
    # sampleCount = int(len(X) - len(Y))
    #
    # mlp = CLib.init(N)
    # mlp = CLib.fit_regression(mlp, X, Y, sampleCount, epochs, alpha)
    #
    # predictions = CLib.predict(mlp, [0, 0], N)
    # print(predictions)
    #
    # predictions = CLib.predict(mlp, [1, 0], N)
    # print(predictions)
    #
    # predictions = CLib.predict(mlp, [0, 1], N)
    # print(predictions)
    #
    # predictions = CLib.predict(mlp, [1, 1], N)
    # print(predictions)

    N = [1, 1]

    X = [
        1,
        2
    ]

    Y = [2, 3]

    sampleCount = 2

    mlp = CLib.init(N)
    mlp = CLib.fit_regression(mlp, X, Y, sampleCount, epochs, alpha)

    predictions = CLib.predict(mlp, [1], N)
    print(predictions)


if __name__ == "__main__":
    main()
