"""
Convolutional neural network architecture.
Author: Chandini, Joselyne, Joel 
Date: 04.28.2023
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU 
from keras import layers 
from keras.models import Model

class CNNModel(Model):
    def create_model(n_char, w, h):
        img = layers.Input(shape=(h, w, 1))  
        conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)  
        mp1 = layers.MaxPooling2D(padding='same')(conv1)  
        conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
        mp2 = layers.MaxPooling2D(padding='same')(conv2)  
        conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
        bn = layers.BatchNormalization()(conv3)  # to improve the stability of model
        mp3 = layers.MaxPooling2D(padding='same')(bn)  

        flat = layers.Flatten()(mp3)  # convert the layer into 1-D

        outs = []
        for i in range(5):  # for 5 letters of captcha
            dens1 = layers.Dense(64, activation='relu')(flat)
            drop = layers.Dropout(0.5)(dens1) 
            res = layers.Dense(n_char, activation='sigmoid')(drop)

            outs.append(res) 

        # Compile model and return it
        model = Model(img, outs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        return model