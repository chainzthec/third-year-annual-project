import numpy as np
import os
from PIL import Image


def dataset_to_matrix(folder_name, x_dim, y_dim):
    folder = "../Dataset/Train/%s" % folder_name
    dataset_matrix = []
    for img in os.listdir(folder):
        img = Image.open("../Dataset/Train/%s/%s" % (folder_name, img))
        resized_img = img.resize((x_dim, y_dim), resample=Image.BILINEAR)
        pix = np.array(resized_img)
        dataset_matrix.append(pix)
    return dataset_matrix


print(dataset_to_matrix("France", 16, 16)[1])


