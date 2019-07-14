#!/usr/bin/env python
# coding=utf-8

#
# Created by Baptiste Vasseur | May 8, 2019
#

import sys
import cv2
import numpy as np
import os
import time

# import MLP.CLibrary as MLP
import MLP.CLibrary as MLP
import Utils


def image_to_array(image):
    list_pixel = []
    for i in range(0, np.size(image, axis=0)):
        for j in range(0, np.size(image, axis=1)):
            for rgb in image[i, j]:
                list_pixel.append(rgb)

    return list_pixel


def start(path, imgSize=(16, 16)):
    validExt = {'jpg', 'png', 'jpeg'}
    XTrain = []
    YTrain = []
    classe = 0
    imgNb = 0
    nbPixPerLine = imgSize[0] * imgSize[1] * 3

    for root, dirList, files in os.walk(path):

        if 0 < len(dirList) < 2:
            print("La dataset doit être composé de 2 classes au minimum !")
            return

        if 0 < len(dirList):
            print("Classes : ", dirList)

            for singleDir in dirList:

                list_possibily = [-1] * len(dirList)
                list_possibily[classe] = 1
                fullpath = os.path.join(root, singleDir)
                filelist = os.listdir(fullpath)

                for filename in filelist:
                    if filename.split(".")[-1].lower() in validExt:
                        filename = os.path.join(fullpath, filename)
                        try:
                            # print("Image : " + filename + " du répertoire : " + singleDir + " : chargé")
                            image = cv2.imread(filename)
                            image = cv2.resize(image, imgSize)
                            pixel = image_to_array(image)
                            XTrain += pixel
                            YTrain += list_possibily
                            imgNb += 1

                        except Exception:
                            continue
                            # print("L'image : " + filename + " du répertoire : " + singleDir + " est inutilisable.")

                print("Dossier " + str(singleDir) + " chargé !")
                classe += 1

    return XTrain, YTrain, imgNb, nbPixPerLine


if __name__ == "__main__":
    inputVal = input("Dataset à entrainer : ")
    filepath = inputVal.rstrip(' ') + '/'.replace("\\ ", ' ')
    xT, yT, sampleCount, inputCountPerSample = start(filepath, (32, 32))

    print("")
    # print(xT)
    # print(yT)
    print("- Taille d'une image : " + str(inputCountPerSample))
    print("- Nombre d'images : " + str(sampleCount))

    print("")

    # N = [3072, 64, 64, 1]
    # epochs = 100
    # alpha = 0.001

    # N = [3072, 64, 64, 1]
    # epochs = 1000
    # alpha = 0.001
    #
    N = [3072, 128, 32, 1]
    epochs = 500
    alpha = 0.01

    start_time = time.time()

    mlpClassif = MLP.init(N)
    mlpClassif = MLP.fit_classification(mlpClassif, xT, yT, sampleCount, epochs, alpha)

    end_time = time.time()

    print("")
    print("- Durée : %s secondes !" % (end_time - start_time))
    print("")

    # Utils.save(mlpClassif, 'MLP', 'MLP_classification_E100_A0001_N3072_64_64_1')
    # Utils.save(mlpClassif, 'MLP', 'MLP_classification_E1000_A0001_N3072_64_64_1.model')
    Utils.save(mlpClassif, 'MLP', 'classification_E500_A001_N3072_256_16_1')
