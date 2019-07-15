#!/usr/bin/env python

#
# Created by Baptiste Vasseur | May 8, 2019
#

import os
import cv2
import numpy as np


def image_to_array(image):
    list_pixel = []
    for i in range(0, np.size(image, axis=0)):
        for j in range(0, np.size(image, axis=1)):
            for rgb in image[i, j]:
                list_pixel.append(rgb)
    return list_pixel


def start(imgSize=(16, 16)):
    validExt = {'jpg', 'png', 'jpeg'}
    XTrain = []
    YTrain = []
    classe = 0
    img_nb = 0
    nbPixPerLine = imgSize[0] * imgSize[1]
    for root, dirList, files in os.walk(filepath):
        if 0 < len(dirList) < 2:
            print("La dataset doit être composé de 2 classes au minimum !")
            return
        print("Classes : ", dirList)
        for singleDir in dirList:
            fullpath = os.path.join(root, singleDir)
            filelist = os.listdir(fullpath)
            for filename in filelist:
                if filename.split(".")[-1].lower() in validExt:
                    filename = os.path.join(fullpath, filename)
                    try:
                        image = cv2.imread(filename)
                        image = cv2.resize(image, imgSize)
                        pixel = image_to_array(image)
                        XTrain.insert(img_nb, pixel)
                        YTrain.append(classe)
                        img_nb = img_nb + 1
                    except ValueError:
                        print("L'image : " + filename + " du répertoire : " + singleDir + " est inutilisable.")
            classe += 1
    return XTrain, YTrain, img_nb, nbPixPerLine


if __name__ == "__main__":
    inputVal = input("Dataset à entrainer : ")
    filepath = inputVal.rstrip(' ') + '/'.replace("\\ ", ' ')
    xT, yT, inputCountPerSample, sampleCount = start((16, 16))

    print(xT)
    print(yT)
    print(inputCountPerSample)
    print(sampleCount)
