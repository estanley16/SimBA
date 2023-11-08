#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:24:42 2023

@author: emma

1. loads the original model trained for disease classification and splits it into the encoder + disease head
2. loads the model from bias pre-training and saves only the head
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
from tensorflow.keras.models import Model


disease_load_name = ['exp138'] 
bias_load_name =    ['exp152']
save_name =         ['exp155']

working_dir = '/home/emma/Documents/SBB/'


#load disease prediction model
orig_model = tf.keras.models.load_model(working_dir + disease_load_name + '/best_model_' + disease_load_name + '.h5')

#get encoder
encoder = Model(inputs = orig_model.input, outputs=orig_model.get_layer('flatten1').output)

#get disease prediction head
disease_head = Model(inputs = orig_model.get_layer('flatten1').output, outputs=orig_model.output)

encoder.save(working_dir + save_name + '/encoder_' + save_name + '.h5')
disease_head.save(working_dir + save_name + '/disease_pred_' + save_name + '.h5')

#bias model
bias_model = tf.keras.models.load_model(working_dir + bias_load_name + '/best_model_' + bias_load_name + '.h5')

#get bias prediction head
bias_head = Model(inputs = bias_model.get_layer('flatten1').output, outputs=bias_model.output)
bias_head.save(working_dir + save_name + '/bias_pred_' + save_name + '.h5')
