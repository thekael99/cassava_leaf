import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
from models.resneXt.functions import Classifier
from tensorflow.keras.models import load_model


class B4_model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(EfficientNetB4(include_top=False, weights=None, input_shape=(299, 299, 3)))
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(5, activation="softmax"))
        self.model.load_weights("./models/EfficientnetB4/B4.h5")

    def predict(self, img):
        img = cv2.resize(img, (299, 299))
        img = np.expand_dims(img, axis=0)
        res = self.model.predict(img)
        return res


class Mobilenet:
    def __init__(self):
        self.model = load_model("./models/mobilenet")

    def predict(self, img):
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = tf.cast(img, tf.float32)
        img = img / 255.
        res = self.model.predict(img)
        return res


class resneXt_model:
    def __init__(self, model_path='./models/resneXt/weights/cnn_res2.pt'):
        self.classifier = Classifier(model_path)

    def predict(self, img):
        pred = self.classifier.pred(img)
        return pred


class Ensemble:
    pass


def load_label_names():
    res = ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy", "Unknown"]
    return res


def assign_label(pred, label):
    return label[np.argmax(pred)]
