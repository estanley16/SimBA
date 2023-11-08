#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:49:53 2023

@author: emma

code for model training with reweighing bias mitigation -- must load datagenerator_3d_reweigh
"""


SEED=1
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
tf.random.set_seed(SEED)
import random
random.seed(SEED)
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from datagenerator_3d_reweigh import DataGenerator
import pickle
import argparse

params = {'batch_size': 4,
        'imagex': 173,
        'imagey': 211,
        'imagez': 155,
        'lr': 1e-4,
        'epochs': 300,
        'kernel_size':(3,3,3),
        'pool_size':(2,2,2),
        'patience': 15,
        }

def cnn3d(inputLayer):

    init = tf.keras.initializers.GlorotUniform(seed=SEED)

    x = Conv3D(filters=32, kernel_size=params['kernel_size'], padding='same', kernel_initializer=init, name='conv1')(inputLayer[0])
    x = BatchNormalization(name='bn1')(x)
    x = Activation('sigmoid', name='act1')(x)
    x = MaxPool3D(pool_size=params['pool_size'], padding='same', name='mp1')(x)

    x = Conv3D(filters=64, kernel_size=params['kernel_size'], padding='same', kernel_initializer=init, name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('sigmoid', name='act2')(x)
    x = MaxPool3D(pool_size=params['pool_size'], padding='same', name='mp2')(x)

    x = Conv3D(filters=128, kernel_size=params['kernel_size'], padding='same', kernel_initializer=init, name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('sigmoid', name='act3')(x)
    x = MaxPool3D(pool_size=params['pool_size'], padding='same', name='mp3')(x)

    x = Conv3D(filters=256, kernel_size=params['kernel_size'], padding='same', kernel_initializer=init, name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('sigmoid', name='act4')(x)
    x = MaxPool3D(pool_size=params['pool_size'], padding='same', name='mp4')(x)

    x = Conv3D(filters=512, kernel_size=params['kernel_size'], padding='same', kernel_initializer=init, name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('sigmoid', name='act5')(x)
    x = MaxPool3D(pool_size=params['pool_size'], padding='same', name='mp5')(x)

    x = AveragePooling3D(padding='same', name='avgpool1')(x)
    x = Dropout(.2, name='dropout1')(x)
    x = Flatten(name='flatten1')(x)
    x = Dense(units=2, activation='softmax', kernel_initializer=init, name="classification")(x)

    return x

    
def compile_model():
    opt = Adam(learning_rate=params['lr'])
    metr = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
    inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
    z = cnn3d([inputA])
    model = Model(inputs=[inputA], outputs=[z])
    model.summary()
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=metr)
    return model
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, help='experiment name', required=True)
    args = parser.parse_args()
    exp_name = args.expname
    
    
    home_dir = '/home/emma/Documents/SBB/'
    working_dir = home_dir + exp_name + '/'
    
    fn_train = working_dir + "train.csv"
    train = pd.read_csv(fn_train)
    train_fpaths = train['filepath'].to_numpy()
    training_generator = DataGenerator(list_IDs=train_fpaths, batch_size=params['batch_size'], dim=(params['imagex'], params['imagey'], params['imagez']), shuffle=True, filename=fn_train)
    
    fn_val = working_dir + "val.csv"
    val = pd.read_csv(fn_val)
    val_fpaths = val['filepath'].to_numpy()
    val_generator = DataGenerator(list_IDs=val_fpaths, batch_size=params['batch_size'], dim=(params['imagex'], params['imagey'], params['imagez']), shuffle=True, filename=fn_val)
    

    model = compile_model()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(working_dir + "best_model_" + exp_name  + ".h5", monitor='val_loss', verbose=2,
                                                             save_best_only=True, include_optimizer=True,
                                                             save_weights_only=False, mode='auto',
                                                             save_freq='epoch')
    
    
    history = model.fit(training_generator, epochs=params['epochs'], validation_data=val_generator,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience']),checkpoint_callback], verbose=2)

    with open(working_dir + exp_name + '_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    model.save(working_dir + "last_model_" + exp_name + ".h5")



if __name__ == '__main__':
    main()
