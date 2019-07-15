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

import Implementation.MLP.MLP as MLP
import Utils


def image_to_array(image):
    list_pixel = []
    for i in range(0, np.size(image, axis=0)):
        for j in range(0, np.size(image, axis=1)):
            for rgb in image[i, j]:
                list_pixel.append(rgb)

    return list_pixel


def start(path, _size):
    validExt = {'jpg', 'png', 'jpeg'}
    XTrain = []
    YTrain = []
    classe = 0
    imgNb = 0
    nbPixPerLine = _size[0] * _size[1] * 3

    for root, dirList, files in os.walk(path):

        if 0 < len(dirList) < 2:
            print("La dataset doit être composé de 2 classes au minimum !")
            return

        if 0 < len(dirList):

            for singleDir in dirList:

                list_possibily = [-1] * len(dirList)
                list_possibily[classe] = 1

                print("Classe :", singleDir, '->', list_possibily)

                fullpath = os.path.join(root, singleDir)
                filelist = os.listdir(fullpath)

                for filename in filelist:
                    if filename.split(".")[-1].lower() in validExt:
                        filename = os.path.join(fullpath, filename)
                        try:
                            # print("Image : " + filename + " du répertoire : " + singleDir + " : chargé")
                            image = cv2.imread(filename)
                            image = cv2.resize(image, _size)
                            pixel = image_to_array(image)
                            XTrain += pixel
                            YTrain += list_possibily
                            imgNb += 1

                        except Exception:
                            continue
                            # print("L'image : " + filename + " du répertoire : " + singleDir + " est inutilisable.")

                print("     > Dossier " + str(singleDir) + " chargé !")
                classe += 1

    return XTrain, YTrain, imgNb, nbPixPerLine


if __name__ == "__main__":
    inputVal = input("Dataset à entrainer : ")
    filepath = inputVal.rstrip(' ') + '/'.replace("\\ ", ' ')

    largeur = input("Largeur de l'image ? (par défaut 32) ")
    largeur = 32 if len(largeur) == 0 else int(largeur)
    size = (largeur, largeur)

    epochs = input("Epochs ? (par défaut 1000) ")
    epochs = 1000 if len(epochs) == 0 else int(epochs)

    alpha = input("Alpha ? (par défaut 0.01) ")
    alpha = 0.01 if len(alpha) == 0 else float(alpha)

    xT, yT, sampleCount, inputCountPerSample = start(filepath, size)
    N = [inputCountPerSample, 64, 64, 2]

    print("")
    print("- Epochs : " + str(epochs))
    print("- Alpha : " + str(alpha))
    print("- Taille d'une image : ", str(_s) + "x" + str(_s), 'x3 -> ', str(inputCountPerSample))
    print("- Nombre d'images : " + str(sampleCount))

    print("")

    start_time = time.time()

    mlpClassif = MLP.init(N)
    mlpClassif = MLP.fit_classification(mlpClassif, xT, yT, sampleCount, epochs, alpha)

    end_time = time.time()

    print("")
    print("- Durée : %s secondes !" % (end_time - start_time))
    print("")

    # Utils.save(mlpClassif, 'MLP', 'classification_E100_A0001_N3072_64_64_2')
    # Utils.save(mlpClassif, 'MLP', 'classification_E1000_A0001_N3072_64_64_2.model')
    # Utils.save(mlpClassif, 'MLP', 'classification_E500_A001_N3072_128_16_2')
    Utils.save(mlpClassif, 'MLP', 'classification_E1000_A001_N3072_64_64_2.model')
