import os

#Load model and predict on data using CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array 
import tensorflow as tf
import pickle
import numpy as np
import pathlib

class ModelPrediction():
    def __init__(self, model_name):
        self.CWD = pathlib.Path(__file__).parent.resolve()
        self._PRODUCTION_DIR = 'production'
        self._MODEL_DIR = os.path.join(self.CWD, self._PRODUCTION_DIR, 'output', 'vgg16_base')
        self._MODEL = None
        self._CLASS_INDEX_PATH = os.path.join(self.CWD, self._PRODUCTION_DIR, 'class_index.pkl')
        self._CLASS_INDEX = None
        self.model_name = model_name

        self._init_model()

    def _init_model(self):
        with open(self._CLASS_INDEX_PATH, 'rb') as file:
            self._CLASS_INDEX = pickle.load(file)
        self._MODEL = self._get_model('vgg16_base', True)

    def _map_pred_to_class(self, pred):
        idx = np.argmax(pred)
        for key, value in self._CLASS_INDEX.items():
            if value == idx:
                return key

    def _get_model(self, name, debug = False):
        model_name = name 
        model = load_model(os.path.join(self._MODEL_DIR))
        print("Loaded model: ", model_name)
        if debug:
            model.summary()
        return model

    def _load_image(self, path):
        #The model expect img input shape to be (batch_size, height, width, channels)
        img = load_img(path, target_size=(32,32,3))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis = 0)
        img_tensor /= 255.0
        return img_tensor

    def _map_pred_to_class(self, pred):
        idx = np.argmax(pred)
        for key, value in self._CLASS_INDEX.items():
            if value == idx:
                return key


    def predict(self, image_path):
        img = self._load_image(image_path)
        pred = self._MODEL.predict(img)
        return  self._map_pred_to_class(pred)
    
