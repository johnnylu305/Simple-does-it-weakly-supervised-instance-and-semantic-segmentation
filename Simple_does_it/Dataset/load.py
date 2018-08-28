import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import tqdm

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True

class Load:
    def __init__(self, is_train, dataset, set_name, label_dir_name, img_dir_name, width, height):
        # training or testing
        self.is_train = is_train
        # get dataset path
        self.dataset_path = dataset
        # get set name
        self.set_name = set_name
        # get label directory name
        self.label_dir_name = label_dir_name
        # get image directory name
        self.img_dir_name = img_dir_name
        self.width = width
        self.height = height
        # image
        self.x = []
        # label
        self.y = []
        # image names
        self.img_names = []

    def load_data(self):
        with open(self.dataset_path + '/' + self.set_name, 'r') as r:
            for img_name in tqdm.tqdm(r, desc = '{:{}}'.format('Load image name', SPACE), unit_scale = UNIT_SCALE):
                #img_name = img_name.rstrip() + '.png'
                img_name = img_name.rstrip()
                # load image
                img = Image.open(self.dataset_path + '/' + self.img_dir_name + '/' + img_name + '.jpg')
                if self.is_train:
                    img = img.resize((self.width, self.height), PIL.Image.LANCZOS)
                    img = np.array(img)
                    self.x.append(img)
                    # load label
                    label = Image.open(self.dataset_path + '/' + self.label_dir_name + '/' + img_name + '.png')
                    label = label.resize((self.width, self.height), PIL.Image.NEAREST)
                    label = np.array(label)
                    # [h, w] to [h, w, 1]
                    label = np.expand_dims(label, axis = 2)
                    self.y.append(label)
                else:
                    img = np.array(img, dtype = np.float16)
                    self.x.append(img)
                    self.img_names.append(img_name)
            r.close()
        
        if self.is_train:
            return np.asarray(self.x, dtype = np.float16), np.asarray(self.y)
        else:
            return np.asarray(self.x), self.img_names





