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
