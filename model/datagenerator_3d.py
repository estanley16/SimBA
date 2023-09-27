#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emma
"""

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.utils import to_categorical

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size, dim, shuffle, filename):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs #numpy array of all filepaths
        self.shuffle = shuffle
        self.filename = filename
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)  # sites

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load based on the ID from csv filter by ID
            dataset = pd.read_csv(self.filename)

            # dataset = dataset[dataset['filename']==ID] #changed it to find filepath instead of ID in case of duplicates
            # path = dataset['filepath'].values

            dataset = dataset[dataset['filepath']==ID]
            path = dataset['filepath'].values

            itk_img = sitk.ReadImage(path)
            np_img = sitk.GetArrayFromImage(itk_img) #1,z,y,x
            np_img = np.swapaxes(np_img,1,3) #1,x,y,z

            #scaling
            scale=np.max(np_img[:])-np.min(np_img[:])
            np_img=(np_img-np.min(np_img[:]))/scale #all values between [0,1]
            np_img = np_img - 0.5 #all values between [-0.5,0.5]


            X[i,] = np.float32(np_img.reshape(self.dim[0], self.dim[1], self.dim[2], 1)) #3D
            # X[i,] = np.float32(np_img.reshape(self.dim[0], self.dim[1], 1)) #2D

            y[i,] = dataset['class_label'].values

        y = to_categorical(y,2)


        return X, y
