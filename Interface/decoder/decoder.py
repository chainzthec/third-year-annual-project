import sys

import cv2
import numpy as np
from PIL import Image

from application.wsgi import application

sys.path.append("..")
import Implementation.Utils as Utils


def launch_traitment(image, model_name):
    print(image)

    file = Image.open(image.file)

    try:
        opencv_image = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {'error': 'Erreur lors de la lecture du fichier !'}

    try:
        image = cv2.resize(opencv_image, application.settings.IMAGE_SIZE)
    except Exception as e:
        return {'error': 'Erreur lors du redimmensionnement du fichier'}

    try:
        pixel = image_to_array(image)
    except Exception as e:
        return {'error': 'Erreur lors du traitement de votre image'}

    try:
        model = load_model(model_name)
    except Exception as e:
        return {'error': 'Erreur lors du chargement du model'}

    print(pixel)
    print(model)

    return {"res": True, 'pixel': pixel}


def load_model(model_name):
    model = load(model_name)
    print(model_name + ' loaded')

    # call load model func from utils.py
    return model


def image_to_array(image):
    list_pixel = []
    for i in range(0, np.size(image, axis=0)):
        for j in range(0, np.size(image, axis=1)):
            for rgb in image[i, j]:
                list_pixel.append(rgb)

    return list_pixel
