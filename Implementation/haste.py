#
# [30/06/2019] Created by Théo Huchard
#

import Implementation.MLP.CLibrary as mlp_clib
import Implementation.RBF.CLibrary as rbf_clib
import Implementation.Rosenblatt.CLibrary as rosenblatt_clib


def train(
        x_train,
        y_train,
        input_count_per_sample,
        sample_count=None,
        epochs=100,
        k=2,
        use_bias=False,
        method="NAIVE_RBF"
):
    if method == "NAIVE_RBF":
        return rbf_clib.naive_rbf_train(x_train, y_train, input_count_per_sample, sample_count, k)
    if method == "RBF":
        return rbf_clib.rbf_train(x_train, y_train, input_count_per_sample, sample_count, epochs, k)  # WIP


def fit(
        model,
        x_train,
        y_train,
        sample_count=None,
        epochs=10,
        alpha=0.01,
        method="MLP_REG"
):
    if method == "MLP_REG":
        return mlp_clib.fit_regression(model, x_train, y_train, sample_count, epochs, alpha)
    if method == "MLP_CLASS":
        return mlp_clib.fit_classification(model, x_train, y_train, sample_count, epochs, alpha)
    if method == "RSB_REG":
        return rosenblatt_clib.fit_classification(rsb_model, X, Y, alpha, epochs)
    if method == "RSB_CLASS":
        return rosenblatt_clib.fit_regression(rsb_model, X, Y)


def predict(
        model,
        sample,
        n=None,
        method="NAIVE_RBF"
):
    if method == "NAIVE_RBF":
        return rbf_clib.naive_rbf_predict(model, sample)
    if method in ("MLP_REG", "MLP_CLASS", "MLP"):
        return mlp_clib.predict(model, sample, n)
    if method == "RSB_REG":
        return rosenblatt_clib.predict_regression(model, sample)
    if method == "RSB_CLASS":
        return rosenblatt_clib.predict_classification(model, sample)


# RBF


# MLP
print("Test MLP")

N = [1, 1]
X2 = [
    1,
    2
]
Y2 = [2, 3]
sampleCount2 = 2
mlp = mlp_clib.init(N)

mlp = fit(mlp, X2, Y2, sampleCount2, epochs=100, alpha=0.01, method="MLP_REG")
print(predict(mlp, [1], N, method="MLP_REG"))

# Rosenblatt
print("Test Rosenblatt")

X3 = [
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
Y3 = [-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1]
input_count_per_sample = int(len(X) / len(Y))
rsb_model = rosenblatt_clib.create_linear_model(input_count_per_sample)
print("\nBefore Classification : ")
rosenblatt_clib.display_matrix(rsb_model, 1, input_count_per_sample + 1)
trained_model_classif = fit(rsb_model, X, Y, alpha=0.001, epochs=5000, method="RSB_CLASS")
print("After Classification : ")
rosenblatt_clib.display_matrix(trained_model_classif, 1, input_count_per_sample + 1)
