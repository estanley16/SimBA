#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:00:55 2023

@author: emma
"""
SEED=1
import numpy as np 
import pandas as pd 
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.layers import Dense


def get_reweigh_params(dataset):
    '''
    from the training data file, get values for calculating sample weights 
    input: 
        -dataset: pandas file of training data, must contain class_label and bias_label columns
    returns: 
        -d: fraction of disease = 1 (disease class) in dataset
        -gamma: fraction of bias = 1 in disease = 1
        -alpha: fraction of bias = 1 in disease = 0 
    '''
    num_total = len(dataset)
    
    
    df_class_D = dataset.loc[dataset['class_label']==1]
    df_class_ND = dataset.loc[dataset['class_label']==0]
    
    num_class_D = len(df_class_D)
    num_class_ND = len(df_class_ND)
    
    d = num_class_D/num_total
    
    gamma = len(df_class_D.loc[df_class_D['bias_label']==1]) / num_class_D
    alpha = len(df_class_ND.loc[df_class_ND['bias_label']==1]) / num_class_ND
    
    return d, gamma, alpha



def calculate_weight(class_label, bias_label, d, gamma, alpha):
    '''
    calculate sample weights based on Calders et. al, Building Classifiers with Independency Constraints (2009)
    https://www.win.tue.nl/~mpechen/publications/pubs/CaldersICDM09.pdf
    input:
        -class_label, bias_label: binary labels for the subject
        -d: fraction of disease = 1 (disease class) in dataset
        -gamma: fraction of bias = 1 in disease = 1
        -alpha: fraction of bias = 1 in disease = 0 
    returns: sample weight for the subject
    '''
    
    if class_label == 1 and bias_label == 1: 
        w = (gamma+alpha)*d/gamma
    
    elif class_label == 0 and bias_label == 1: 
        w = (gamma+alpha)*(1-d)/alpha
        
    elif class_label == 1 and bias_label == 0: 
        w = (2-gamma-alpha)*d/(1-gamma)
        
    elif class_label == 0 and bias_label == 0: 
        w = (2-gamma-alpha)*(1-d)/(1-alpha)
    
    return w



def chop_model(base_model, layer_name=None):
    '''
    chop off bottom part of a model
    
    if layer_name is specified, keep up to that layer (inclusive)
    if layer_name not specified, just remove last layer
    '''
    if layer_name is not None:     
        layer_output = base_model.get_layer(layer_name).output
        new_model = tf.keras.models.Model(inputs=base_model.input, outputs=layer_output)
    else:
        new_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return new_model


