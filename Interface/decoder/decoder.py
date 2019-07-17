import sys
import os
import numpy as np
from PIL import Image
from application.wsgi import application
import cv2

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import Utils
import Implementation.MLP.MLP as MLP
import Implementation.Linear.Linear as Linear


def launch_traitment(image, model_name):
    file = Image.open(image.file)

    try:
        opencv_image = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return {'error': 'Erreur lors de la lecture du fichier !'}

    try:
        image = cv2.resize(opencv_image, (32, 32))
<<<<<<< HEAD
    except Exception as e:
        return {'error': 'Erreur lors du redimmensionnement du fichier'}

    try:
        pixel = image_to_array(image)
    except Exception as e:
        return {'error': 'Erreur lors du traitement de votre image'}

    # try:
    model, algo_name = load_model(model_name)
    # except Exception as e:
    #     return {'error': 'Erreur lors du chargement du model'}

    res = Utils.predict(model, algo_name, pixel)
=======
    except Exception as e:
        return {'error': 'Erreur lors du redimmensionnement du fichier'}

    try:
        pixel = image_to_array(image)
    except Exception as e:
        return {'error': 'Erreur lors du traitement de votre image'}

    try:
        model, model_name = load_model(model_name)
    except Exception as e:
        return {'error': 'Erreur lors du chargement du model'}

    res = Utils.predict(model, model_name, pixel)
>>>>>>> 7044ecb369e2c2312ec8a9d4f4c0146bdbeeae00

    print('\n\n----------------')
    print('Predicted image size :', len(pixel), "pixels")
    print("Result :", res)
    print('----------------\n\n')
    # print(pixel)
    # print(model)

<<<<<<< HEAD
    classe = "?"
    if algo_name.upper() == "MLP":

=======
    if algo_name.upper() == "MLP":

        classe = "?"
>>>>>>> 7044ecb369e2c2312ec8a9d4f4c0146bdbeeae00
        maxindex = res.index(max(res))
        if maxindex == 0:
            classe = "France"
        elif maxindex == 1:
            classe = "Italie"
        elif maxindex == 2:
            classe = "Congo"

    elif algo_name.upper() == "LINEAR":
<<<<<<< HEAD
        if res < 0:
            classe = "France"
        else:
            classe = "Italie"
=======
        classe = "ok"
>>>>>>> 7044ecb369e2c2312ec8a9d4f4c0146bdbeeae00

    return {"res": True, 'result': res, 'classe': classe}


def load_model(model_name):
    model = Utils.load(model_name)
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
