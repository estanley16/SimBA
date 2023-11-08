#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:07:50 2023

@author: emma

Protected group models: fine tune model on each bias group 
based on Puyol-Anton et al,"Fairness in Cardiac MR Image Analysis: An Investigation of Bias Due to Data Imbalance in Deep Learning Based Segmentation" (2021)

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
from datagenerator_3d import DataGenerator
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


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, help='experiment name to load model for', required=True)
    parser.add_argument('--bias_label', type=int, help='bias label to fine tune on, 0 or 1', required=True)
    args = parser.parse_args()
    
    exp_name = args.expname
    bias_label = args.bias_label 
    
    home_dir = '/home/emma/Documents/SBB/'
    working_dir = home_dir + exp_name + '/'
    
    
    fn_train = working_dir + "train.csv"
    train = pd.read_csv(fn_train)
    train = train.loc[train['bias_label']==bias_label] #slice out only specified bias label 
    train_fpaths = train['filepath'].to_numpy()
    training_generator = DataGenerator(train_fpaths, params['batch_size'],(params['imagex'], params['imagey'], params['imagez']), True, fn_train)
    
    fn_val = working_dir + "val.csv"
    val = pd.read_csv(fn_val)
    val = val.loc[val['bias_label']==bias_label] #slice out only specified bias label 
    val_fpaths = val['filepath'].to_numpy()
    val_generator = DataGenerator(val_fpaths, params['batch_size'],(params['imagex'], params['imagey'], params['imagez']), False, fn_val)
    
    model = tf.keras.models.load_model(working_dir + 'pretrained_model_' + exp_name + '.h5') #import pretrained model 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(working_dir + 'best_model_finetuned_' + exp_name + '_biaslabel' + str(bias_label) + '_.h5', monitor='val_loss', verbose=2,
                                                              save_best_only=True, include_optimizer=True,
                                                              save_weights_only=False, mode='auto',
                                                              save_freq='epoch')
    
    
    history = model.fit(training_generator, epochs=params['epochs'], validation_data=val_generator,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience']), checkpoint_callback], verbose=2)

    with open(working_dir + exp_name + '_biaslabel' + str(bias_label)  + '_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    model.save(working_dir + 'last_model_finetuned_' + exp_name + '_biaslabel' + str(bias_label)  + '_.h5')




if __name__ == '__main__':
    main()
