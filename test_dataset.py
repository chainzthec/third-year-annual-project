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


def start(_path, _size, _model, _algo_name):
    validExt = {'jpg', 'png', 'jpeg'}
    classe = 0
    imgNb = 0
    valid = 0
    invalid = 0
    nbPixPerLine = _size[0] * _size[1] * 3

    for root, dirList, files in os.walk(_path):

        if 0 < len(dirList) < 2:
            print("La dataset doit être composé de 2 classes au minimum !")
            return

        if 0 < len(dirList):

            for singleDir in dirList:

                YTrain = [-1] * len(dirList)
                YTrain[classe] = 1

                print("Classe :", singleDir, '->', YTrain)

                fullpath = os.path.join(root, singleDir)
                filelist = os.listdir(fullpath)

                for i, filename in enumerate(filelist):
                    if filename.split(".")[-1].lower() in validExt:
                        filename = os.path.join(fullpath, filename)
                        try:
                            image = cv2.imread(filename)
                            image = cv2.resize(image, _size)
                            xToPredict = image_to_array(image)

                            res = Utils.predict(_model, _algo_name, xToPredict)
                            indexRes = res.index(max(res))

                            YTrainIndex = YTrain.index(max(YTrain))
                            if indexRes == YTrainIndex:
                                print('#' + str(i), '> filename', filename, '> indexRes', indexRes, '-',
                                      'YTrainIndex', YTrainIndex, '-> GOOD')
                                valid += 1
                            else:
                                print('#' + str(i), '> filename', filename, '> indexRes', indexRes, '-',
                                      'YTrainIndex', YTrainIndex, '-> BAD')
                                invalid += 1

                            imgNb += 1

                        except Exception:
                            continue

                print("     > Dossier " + str(singleDir) + " chargé !")
                classe += 1

    return imgNb, nbPixPerLine, valid, invalid


if __name__ == "__main__":
    inputVal = input("Dataset à tester : ")
    filepath = inputVal.rstrip(' ') + '/'.replace("\\ ", ' ')

    start_time = time.time()

    # MLP_classification_E500_A001_N3072_128_16_2.model
    _model_name = input('Modèle à charger ? : ')
    model, algo_name = Utils.load(_model_name)

    largeur = input("Largeur de l'image ? (par défaut 32) ")
    largeur = 32 if len(largeur) == 0 else int(largeur)
    size = (largeur, largeur)

    sampleCount, inputCountPerSample, valid, invalid = start(filepath, size, model, algo_name)

    print()
    print()
    print("- Taille d'une image : " + str(inputCountPerSample))
    print("- Nombre d'images : " + str(sampleCount))

    print("- Prédiction correct : " + str(valid) + "/" + str(sampleCount) + " soit " + str((100 * valid / sampleCount)) + "%")
    print("- Prédiction incorrect : " + str(invalid) + "/" + str(sampleCount) + " soit " + str((100 * invalid / sampleCount)) + "%")

    end_time = time.time()

    print("")
    print("- Durée : %s secondes !" % (end_time - start_time))
    print("")

