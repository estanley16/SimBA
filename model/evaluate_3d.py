#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emma
"""

import pickle
from datagenerator_3d import DataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
import argparse
seed(1)
tf.random.set_seed(1)
random.seed(1)


def model_eval(y_test, y_pred_raw):
    '''
    converts raw softmax outputs to dataframe with true labels and predicted labels
    '''
    y_pred = np.argmax(y_pred_raw, axis=1).reshape(-1,1)
    y_pred = y_pred.astype(int)
    y_test = y_test.to_frame()
    y_test = y_test.rename(columns={'class_label': 'ground_truth'})
    y_test['preds'] = y_pred
    y_test['preds_raw_0'] = y_pred_raw[:,0]
    y_test['preds_raw_1'] = y_pred_raw[:,1]
    return y_test


def compute_metrics(df, save_dir, label):
    '''
    get confusion matrix and list of metrics from df with predictions and labels
    '''
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values
    y_pred_raw = df['preds_raw_1'].values

    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print(cm)
    cm_df = pd.DataFrame(cm,
            index = ['ND','D'],
            columns = ['ND','D'])
      #Plotting the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, cmap="Blues", annot=True,fmt='.2f', vmin=0, vmax=1.0, center=0.5,cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(save_dir + label + '_cm.png', bbox_inches='tight', dpi = 800)
    plt.clf()

    ac=accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    ap = average_precision_score(y_test, y_pred_raw)

    fpr = 1-spec

    auc = roc_auc_score(y_test, y_pred_raw)

    return [ac, sens, spec, fpr, ap, auc]



def plot_training_curves(history, save_dir, validation = None):
    # summarize history for accuracy
    plt.plot(history['accuracy'])

    if validation is not None:
        plt.plot(history['val_accuracy'])
        plt.legend(['train', 'val'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(save_dir + '_acc.png', bbox_inches='tight', dpi = 800)
    # plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history['loss'])

    if validation is not None:
        plt.plot(history['val_loss'])
        plt.legend(['train', 'val'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(save_dir + '_loss.png', bbox_inches='tight', dpi = 800)
    # plt.show()
    plt.clf()
    return


def main():

    #------------------- parameters and directory paths ----------------------------#
    params = {'batch_size': 1,
            'imagex': 173,
            'imagey': 211,
            'imagez': 155,
            }


    #use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
    args = parser.parse_args()
    exp_name = args.exp_name

    model_name = 'best_model_' + exp_name + '.h5'

    home_dir = '/home/emma/Documents/SBB/'
    working_dir = home_dir + exp_name + '/'

    #------------------- ------------------------------ ----------------------------#

    #test data and generator
    fn_test = working_dir + "test.csv"
    test = pd.read_csv(fn_test, index_col = 'filepath')
    test_fpaths=test.index.to_numpy()
    test_generator=DataGenerator(test_fpaths, params['batch_size'],
                                 (params['imagex'], params['imagey'], params['imagez']), False, fn_test)


    # load model for evaluation
    model=tf.keras.models.load_model(working_dir + model_name)
    model.trainable=False


    # Evaluate model on test set
    y_test = test['class_label']
    y_pred_raw = model.predict(test_generator)
    preds = model_eval(y_test, y_pred_raw)
    df = pd.merge(preds, test, left_index=True, right_index=True)

    #create one-hot encoded columns for TP, TN, FP, FN
    df['TP'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 1) & (row['preds']==1)) else 0, axis=1)
    df['TN'] = df.apply(lambda row: 1 if ((row['ground_truth']== 0) & (row['preds'] ==0)) else 0, axis=1)
    df['FP'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 0) & (row['preds'] ==1)) else 0, axis=1)
    df['FN'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 1) & (row['preds'] ==0)) else 0, axis=1)

    #save file with predictions and information about all the test data
    df.to_csv(working_dir + 'preds_' + exp_name + '.csv')


    df_B1 = df.loc[df['bias_label']==1]
    df_B0 = df.loc[df['bias_label']==0]

    #generate file with performance metrics
    metrics = compute_metrics(df, save_dir=working_dir, label='Agg') #performance metrics computed on all data
    metrics_B1 = compute_metrics(df_B1, save_dir=working_dir, label='B1') #performance metrics computed on group w/ bias effect
    metrics_B0 = compute_metrics(df_B0, save_dir=working_dir, label='B0') #performance metrics computed on group w/o bias effect

    metrics_df = pd.DataFrame(['Acc', 'Sens', 'Spec', 'FPR', 'AP', 'AUROC'], columns=['metrics'])
    metrics_df = metrics_df.set_index('metrics')

    metrics_df['Aggregate'] = metrics
    metrics_df['bias_label_1'] = metrics_B1
    metrics_df['bias_label_0'] = metrics_B0
    metrics_df.to_csv(working_dir + 'metrics_' + exp_name + '.csv')


    #plot history
    with open(working_dir + exp_name + '_history.pkl', 'rb') as f:
        hst = pickle.load(f)

    plot_training_curves(hst, working_dir + exp_name, validation=True)


if __name__ == '__main__':
    main()
