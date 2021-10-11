import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4

import json


class B4_model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(EfficientNetB4(include_top=False, weights=None, input_shape=(224, 224, 3)))
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(5, activation="softmax"))
        self.model.load_weights("./models/B4_224/B4.h5")

    def predict(self, img):
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        res = self.model.predict(img)
        return res


class Ensemble:
    pass


def load_label_names():
    with open("label_num_to_disease_map.json") as f:
        res = json.load(f)
    return list(res.values())


def assign_label(pred, label):
    return label[np.argmax(pred)]
