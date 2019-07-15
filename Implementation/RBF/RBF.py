#
# Created by Théo Huchard on 2019-06-30.
#

import CLibrary as CLib

X = [
    0.13984698, 0.41485388,
    0.28093573, 0.36177096,
    0.25704393, 0.97695092,
    0.05471647, 0.8640708,
    0.91900274, 0.95617945,
    0.1753089, 0.67689523,
    0.25784674, 0.12366917,
    0.97495302, 0.01277128,
    0.08287882, 0.94833339,
    0.39418121, 0.7978936
]

Y = [
    0.46119306,
    0.78636786,
    0.2617359,
    0.25985246,
    0.28554652,
    0.57842217,
    0.35202585,
    0.11248387,
    0.72196561,
    0.60782134
]

print("Test")

inputCountPerSample = int(len(X) / len(Y))
sampleCount = 2
model = CLib.naive_rbf_train(X, Y, inputCountPerSample, sampleCount, 100)
print(CLib.naive_rbf_predict(model, [0.28093573, 0.36177096]))

# def naive_rbf_train(X,Y,inputCountPerSample,sampleCount, epochs = 100, k = 2, useBias = False):
