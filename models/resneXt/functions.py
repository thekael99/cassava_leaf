# import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
# from speed_classify.model import Network
from models.resneXt.resneXt import ResNeXt
import torch
# import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from torch import nn


class Classifier():
    def __init__(self, model_path):
        self.sm = nn.Softmax(dim=1)
        self.device = torch.device("cpu")
        lb = pd.read_csv('models/resneXt/label.csv', header=None)
        self.classes = lb.values[:, 0]
        nb_classes = len(self.classes)
        self.img_size = (416, 416)
        # define model
        # self.model = Network(nb_classes).to(device)
        self.model = ResNeXt(4, 32, [3, 4, 6, 3], nb_classes).to(self.device)
        pre_trained = torch.load(model_path, self.device)  # 'cnn_arc.pt'
        self.model.load_state_dict(pre_trained)

        # port to model to gpu if you have gpu
        self.model = self.model.to(self.device)
        self.model.eval()

    def pred(self, img_raw):
        img_rgb = cv2.resize(img_raw, self.img_size)
        # convert from RGB img to gray img
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        # normalize img from [0, 255] to [0, 1]
        img_rgb = img_rgb / 255
        img_rgb = img_rgb.astype('float32')
        img_rgb = img_rgb.transpose(2, 0, 1)

        # convert image to torch with size (1, 1, 48, 48)
        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        with torch.no_grad():
            img_rgb = img_rgb.to(self.device)
            # print("type: " + str(numb), type(img_rgb))
            y_pred = self.model(img_rgb)
            print("y_pred", y_pred)
            y_pred = self.sm(y_pred)
            # _, pred = torch.max(y_pred, 1)
            # print("pred", pred)

            # pred = pred.data.cpu().numpy()
            # pred = y_pred.data.cpu().numpy()
            # print("2nd", second_time - fist_time)
            # print("predict: " +str(numb), pred)
            # class_pred = self.classes[pred[0]]

        return y_pred


class Predict():
    def __init__(self, model_path):
        self.classifier = Classifier(model_path)  # 'weights_19.9/cnn_res2.pt'

    def pred_one_image(self, path=None, img=None):
        if path != None:
            img_raw = cv2.imread(path)
        else:
            img_raw = img

        output = self.classifier.pred(img_raw)
        return output
