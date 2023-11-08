#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Adversarial training to unlearn bias label from disease classification
-> Adapted from Raissa/Vedant's code for unlearning scanner from Parkinson's disease prediction,
based on "Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal" Dinsdale et. al

"""

SEED=1
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
tf.random.set_seed(SEED)
from numpy.random import seed
seed(SEED)
import random
random.seed(SEED)
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from datagenerator_3d_unlearning import DataGenerator
import argparse
import time

EPOCHS = 30
alpha=1
beta=1
#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help='experiment name', required=True)

args = parser.parse_args()


exp_name = args.exp_name

params = {'batch_size': 4,
        'imagex': 173,
        'imagey': 211,
        'imagez': 155,
        'lr': 1e-4,
        'epochs': 30,
        'kernel_size':(3,3,3),
        'pool_size':(2,2,2),
        'patience': 15,
        'bias_groups': 2,
        }


home_dir = '/home/emma/Documents/SBB/'
working_dir = home_dir + exp_name + '/'


# pretrained models
encoder = tf.keras.models.load_model(working_dir + 'encoder_' + exp_name + '.h5')
classifier_D = tf.keras.models.load_model(working_dir + 'disease_pred_' + exp_name + '.h5') #disease classifier
classifier_B = tf.keras.models.load_model(working_dir + 'bias_pred_' + exp_name + '.h5') #bias classifier

encoder.trainable = True
classifier_D.trainable = True
classifier_B.trainable = True


# optimizers
optimizer_encoder = Adam(learning_rate=params['lr'])
optimizer_D = Adam(learning_rate=params['lr'])
optimizer_B = Adam(learning_rate=params['lr'])

encoder.compile(optimizer=optimizer_encoder)
classifier_D.compile(optimizer=optimizer_D)
classifier_B.compile(optimizer=optimizer_B)



train_loss_B = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
val_loss_B = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_B = tf.keras.metrics.CategoricalAccuracy()
val_acc_B = tf.keras.metrics.CategoricalAccuracy()

train_loss_D = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
val_loss_D = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_acc_D = tf.keras.metrics.CategoricalAccuracy()
val_acc_D = tf.keras.metrics.CategoricalAccuracy()


def confusionLoss(logits_B, batch_size):
    log_logits = tf.math.log(logits_B)
    sum_log_logits = tf.math.reduce_sum(log_logits)
    #norm = sum_log_logits/batch_size
    return -1*sum_log_logits / (batch_size * params['bias_groups'])
    #return -1*norm


# Dataset generator for disease classification
fn_train = working_dir + "train.csv"
train = pd.read_csv(fn_train)
train_fpaths = train['filepath'].to_numpy()


fn_val = working_dir + "val.csv"
val = pd.read_csv(fn_val)
val_fpaths = val['filepath'].to_numpy()


# train step for disease classifier
@tf.function
def train_step( X, y_D, y_B):
    ###################################################
    # FIRST STEP MAIN TASK - Disease
    classifier_B.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_D = classifier_D(logits_enc, training=True)
        train_loss_D = train_loss_D(tf.one_hot(y_D, 2), logits_D)
        train_acc_D.update_state(tf.one_hot(y_D, 2), logits_D)

    # compute gradient
    grads = tape.gradient(train_loss_D, [encoder.trainable_weights, classifier_D.trainable_weights])

    # update weights
    optimizer_encoder.apply_gradients(zip(grads[0], encoder.trainable_weights))
    optimizer_D.apply_gradients(zip(grads[1], classifier_D.trainable_weights))
    ###################################################
    # SECOND STEP DOMAIN TASK - BIAS
    encoder.trainable = False
    classifier_D.trainable = False
    classifier_B.trainable = True
    with tf.GradientTape() as tape:
        #logits_enc = encoder(X, training=False)
        logits_B = classifier_B(logits_enc, training=True)
        train_loss_B = alpha * train_loss_B(tf.one_hot(y_B, 2), logits_B)
        train_acc_B.update_state(tf.one_hot(y_B, 2), logits_B)

    # compute gradient
    grads = tape.gradient(train_loss_B, classifier_B.trainable_weights)

    # update weights
    optimizer_B.apply_gradients(zip(grads, classifier_B.trainable_weights))
    encoder.trainable = True
    classifier_D.trainable = True
    ###################################################
    # THIRD STEP
    classifier_D.trainable = False
    classifier_B.trainable = False
    with tf.GradientTape() as tape:
        logits_enc = encoder(X, training=True)
        logits_B = classifier_B(logits_enc, training=False)
        confusion_loss = alpha * confusionLoss(logits_B, params['batch_size'])

    # compute gradient
    grads = tape.gradient(confusion_loss, encoder.trainable_weights)

    # update weights
    optimizer_encoder.apply_gradients(zip(grads, encoder.trainable_weights))
    classifier_D.trainable = True
    classifier_B.trainable = True
    ###################################################

    train_loss = train_loss_D + alpha * train_loss_B + alpha * confusion_loss

    return train_loss, logits_D, logits_B


# test step for disease classifier
@tf.function
def test_step( X, y_D, y_B):

    val_logits_enc = encoder(X, training=False)
    val_logits_D = classifier_D(val_logits_enc, training=False)
    val_logits_B = classifier_B(val_logits_enc, training=False)
    val_acc_D.update_state(tf.one_hot(y_D, 2), val_logits_D)
    val_acc_B.update_state(tf.one_hot(y_B, 2), val_logits_B)

    val_loss_D = val_loss_D(tf.one_hot(y_D, 2), val_logits_D)
    val_loss_B = val_loss_B(tf.one_hot(y_B, 2), val_logits_B)
    val_confusion_loss = confusionLoss(val_logits_B, params['batch_size'])

    # Compute the loss value
    val_loss =  val_loss_D + alpha * val_loss_B + alpha * val_confusion_loss

    return val_loss, val_logits_D, val_logits_B

val_losses=[]
max_acc_D=0
max_acc_D2=0
min_acc_B=0
acc_D=[]
acc_B=[]
patience = 5
performance_D = []
performance_B = []

####################################################################################################################

# training Disease classifier
for epoch in range(params['epochs']):
    # training
    training_generator = DataGenerator(train_fpaths, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_train)
    t1 = time.time()
    total_train_loss=0
    total_val_loss=0
    for batch in range(training_generator.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_D, y_B = training_generator.__getitem__(step_batch)
        train_loss, logits_D, logits_B = train_step( X, y_D, y_B)
        print('\nBatch '+str(batch+1)+'/'+str(training_generator.__len__()))
        print("LOSS D -->", train_loss)
        # for _ in range(params['batch_size']):
        #     print("LOGITS D -->", logits_D[_])
        #     print("ACTUAL D -->", y_D[_])
        total_train_loss+=train_loss

    print("Training loss over epoch: %.4f" % (float(total_train_loss/training_generator.__len__()),))
    train_D_acc = train_acc_D.result()
    print("Training acc D over epoch: %.4f" % (float(train_D_acc),))

    train_B_acc = train_acc_B.result()
    print("Training acc B over epoch: %.4f" % (float(train_B_acc),))


    t2 = time.time()
    template = 'TRAINING - ETA: {} - epoch: {}\n'
    print(template.format(round((t2-t1)/60, 4), epoch+1))


    # validation
    val_generator = DataGenerator(val_fpaths, params['batch_size'], (params['imagex'], params['imagey'], params['imagez']), True, fn_val)
    t3 = time.time()
    acc_ep_D=0
    acc_ep_B=0
    for batch in range(val_generator.__len__()):
        step_batch = tf.convert_to_tensor(batch, dtype=tf.int64)
        X, y_D, y_B = val_generator.__getitem__(step_batch)
        val_loss, val_logits_D, val_logits_B = test_step(X, y_D, y_B)
        print('\nBatch '+str(batch+1)+'/'+str(val_generator.__len__()))
        print("VAL LOSS D -->", val_loss)
        # for _ in range(params['batch_size']):
        #     print("LOGITS D -->", val_logits_D[_])
        #     print("ACTUAL D -->", y_D[_])
        total_val_loss+=val_loss

    print("Validation loss over epoch: %.4f" % (float(total_val_loss/val_generator.__len__()),))

    val_D_acc = val_acc_D.result()
    print("Validation acc D over epoch: %.4f" % (float(val_D_acc),))

    val_B_acc = val_acc_B.result()
    print("Validation acc B over epoch: %.4f" % (float(val_B_acc),))

    # print(norm_ep_val_loss)
    val_losses.append(np.around(float(total_val_loss/val_generator.__len__()),4))
    acc_D.append(np.around(float(val_D_acc),4))
    acc_B.append(np.around(float(val_B_acc),4))

    t4 = time.time()
    template = 'VALIDATION - ETA: {} - epoch: {}\n'
    print(template.format(round((t4-t3)/60, 4), epoch+1))

    # learning rate scheduling
    optimizer_encoder = Adam(learning_rate=params['lr'])
    optimizer_D = Adam(learning_rate=params['lr'])

    # Reset training metrics and loss at the end of each epoch
    train_acc_D.reset_states()
    val_acc_D.reset_states()

    train_acc_B.reset_states()
    val_acc_B.reset_states()

    #save every epoch
    encoder.save(working_dir + 'un_encoder_'+exp_name+'_epoch' + str(epoch) + '.h5')
    classifier_D.save(working_dir + 'D_classifier_'+exp_name+ '_epoch' + str(epoch) + '.h5')
    classifier_B.save(working_dir + 'un_B_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')

    #early stopping
    if(len(acc_D)==1): #first epoch
        performance_D.append(0)
        performance_B.append(0)
        max_acc_D=acc_D[-1]
        max_acc_D2=acc_D[-1]
        min_acc_B=acc_B[-1]
        encoder.save(working_dir + 'best_un_encoder_'+exp_name+'_epoch' + str(epoch) + '.h5')
        classifier_D.save(working_dir + 'best_un_D_classifier_'+exp_name+ '_epoch' + str(epoch) + '.h5')
        classifier_B.save(working_dir + 'best_un_B_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')
        print("Saving model epoch: "+str(epoch))
    else:
        if acc_D[-1] > max_acc_D and acc_B[-1]< min_acc_B: #improved after last epoch
            performance_D.append(0)
            performance_B.append(0)
            max_acc_D=acc_D[-1]
            min_acc_B=acc_B[-1]
            encoder.save(working_dir + 'best_un_encoder_'+exp_name+'_epoch' + str(epoch) + '.h5')
            classifier_D.save(working_dir + 'best_un_D_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')
            classifier_B.save(working_dir + 'best_un_B_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')
            print("Saving model epoch: "+str(epoch))
        elif acc_D[-1] > max_acc_D2 and acc_B[-1] > min_acc_B: #only D improved
            performance_D.append(1)
            performance_B.append(1)
            max_acc_D2=acc_D[-1]
            encoder.save(working_dir + 'high_un_encoder_'+exp_name+'_epoch' + str(epoch) + '.h5')
            classifier_D.save(working_dir + 'high_un_D_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')
            classifier_B.save(working_dir + 'low_un_B_classifier_'+exp_name+'_epoch' + str(epoch) + '.h5')
            print("Saving model epoch: "+str(epoch))
        else:
            performance_D.append(1)
            performance_B.append(1) #did not improve
        if len(performance_D) > patience:
            if(sum(performance_D[-patience:])==patience): # if the last 10 performances did not improve
                print('Early stopping. No improvement in validation loss in epoch: '+str(epoch))
                break

####################################################################################################################
print(val_losses)
print(acc_D)
print(acc_B)
print(performance_D)
print(performance_B)

####################################################################################################################
