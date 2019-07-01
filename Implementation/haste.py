#
# [30/06/2019] Created by Théo Huchard
#

import MLP.CLibrary as mlp_clib
import RBF.CLibrary as rbf_clib
import Rosenblatt.CLibrary as rosenblatt_clib


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


