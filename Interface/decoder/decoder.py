import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import Implementation.Utils as Utils
from application.settings import BASE_DIR
from application.wsgi import application


def launchTraitment(image, modelName):
    print(image)

    file = Image.open(image.file)

    try:
        opencvImage = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {'error': 'Erreur lors de la lecture du fichier !'}

    try:
        image = cv2.resize(opencvImage, application.settings.IMAGE_SIZE)
    except Exception as e:
        return {'error': str(e)}
        # return {'error': 'Erreur lors du redimmensionnement du fichier'}

    pixel = image_to_array(image)
    model = loadModel(modelName)

    print(pixel)
    print(model + ' loaded')

    return {"res": True, 'pixel': pixel}


def loadModel(modelName):
    return modelName


def image_to_array(image):
    list_pixel = []
    for i in range(0, np.size(image, axis=0)):
        for j in range(0, np.size(image, axis=1)):
            for rgb in image[i, j]:
                list_pixel.append(rgb)

    return list_pixel
