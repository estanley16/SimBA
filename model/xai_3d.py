#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emma
"""

import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datagenerator_3d import DataGenerator
import pandas as pd
import keras.backend as K
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom
import argparse

# -----------------------------------------------------

def subsample(array, num):
    """
    function for subsampling ID arrays to a constant value
    allieviate memory requirements and ensure both subgroups and sexes have the same amnt of representation
    convert np array back to pandas series in order to sample easily with a random seed, then back to np array
    """
    series = pd.Series(array)
    series_sample = series.sample(n=num, random_state=66)
    array_sample = series_sample.to_numpy()
    return array_sample


def vanilla_grad(image, model, class_index):

    '''
    computes vanilla saliency map
    inputs:
        model - model to compute saliency maps with. must use softmax activation in the classification head
        img - numpy array of image (without first batch dimension)
        class_index - desired class label to visualize saliency maps with
    '''
    #add dimension to represent batch size dimension in model input shape
    image = np.expand_dims(image, axis=0)

    linear_model = tf.keras.models.Model(inputs=model.input,
                          outputs=model.output)
    linear_model.layers[-1].activation = tf.keras.activations.linear


    image = tf.convert_to_tensor(image)

    #GradientTape allows for computations to be tracked
    with tf.GradientTape() as tape:
        tape.watch(image) #track the image with GradientTape
        prediction = linear_model(image) #forward pass of the image through the model to compute prediction
        loss = prediction[:, class_index]


    # Get the gradient of the prediction w.r.t to the input image.
    gradient = tape.gradient(loss, image)

    # take maximum across the channel axis
    gradient = tf.reduce_max(gradient, axis=-1)

    # convert to numpy from eagertensor
    gradient = tf.abs(gradient).numpy()

    return gradient



def smoothgrad(image, model, class_index, n_maps = 20, sigma = 0.2, normalize=True):
    '''
    inputs:
        model: trained model to calculate saliency map with (shape: (x_dim, y_dim, 1))
        image: image to generate saliency map for
        n_maps: number of saliency maps to average for smoothgrad
        sigma: standard deviation of gaussian noise applied to saliency n_maps
        normalize: whether to normalize between 0 and 1
    returns: smoothgrad saliency map

    '''

    if n_maps > 1:

        smap = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        for n in range(n_maps):
            gauss_noise = np.random.normal(0, sigma, image.shape)
            noisy_image = image + gauss_noise
            sg_map = vanilla_grad(noisy_image, model, class_index)
            sg_map = np.squeeze(sg_map, axis=0)
            smap += sg_map

        smap = smap/n_maps

    else:
        smap = vanilla_grad(model, image)


    if normalize:
        # normalize between 0 and 1
        min_val, max_val = np.min(smap), np.max(smap)
        smap_norm = (smap - min_val) / (max_val - min_val + K.epsilon())
        return smap_norm

    else:
        return smap



def gradcam(img, model, layer_name, class_index):

    '''
    inputs:
        model - model to compute saliency maps with. must use softmax activation in the classification head
        img - numpy array of image (without first batch dimension)
        layer_name - convolutional layer with desired activation maps to visualize
        class_index - desired class label to visualize saliency maps with
        normalize - whether to normalize values between 0 and 1
        guided - whether to eliminate negative values in the backpropagation

    references:
        - https://www.statworx.com/en/content-hub/blog/car-model-classification-3-explainability-of-deep-learning-models-with-grad-cam/
        - https://github.com/sicara/tf-explain/blob/master/tf_explain/core/grad_cam.py
    '''
    #add dimension to represent batch size dimension in model input shape
    img = np.expand_dims(img, axis=0)


    # Gradient model, outputs tuple with [output of conv layer, output of classification head]
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(layer_name).output, model.output])


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_index]

    # Output of conv layer (removing batch dim)
    output = conv_outputs[0]

    # Gradients of loss wrt. conv layer (removing batch dim)
    grads = tape.gradient(loss, conv_outputs)[0]

    #2D input
    if len(grads.shape) == 3: #2D input
        # Average weight of filters in conv layer
        weights = tf.reduce_mean(grads, axis=(0, 1))

        # compute sum of (conv filters x gradient weights)
        cam = tf.reduce_sum(tf.multiply(weights,output), axis=-1)

        # Rescale to original image size and min-max scale
        cam = cv2.resize(cam.numpy(), (img.shape[1], img.shape[2]))
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + K.epsilon())

    #3D input
    else: #len(grads.shape) == 4
        # Average weight of filters in conv layer
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))

        # compute sum of (conv filters x gradient weights)
        cam = tf.reduce_sum(tf.multiply(weights,output), axis=-1)

        # Rescale to original image size and min-max scale
        # cam = cv2.resize(cam.numpy(), (img.shape[1], img.shape[2], img.shape[3]))
        cam = zoom(cam, (img.shape[1]/cam.shape[0], img.shape[2]/cam.shape[1], img.shape[3]/cam.shape[2])) #cv2 cant handle 3 dimensions, use scipy instead
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + K.epsilon())

    return heatmap



def visualize_overlay(img, smap, ID, save=False):
    img = (img - img.min()) / (img.max() - img.min() + K.epsilon())
    img = img.reshape(img.shape[1:])
    plt.imshow(img, cmap=plt.cm.gray)
    plt.imshow(smap, cmap=plt.cm.viridis, alpha=.75)
    plt.axis('off')
    plt.show()


def main():

    #------------------- parameters and directory paths ----------------------------#

    #use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
    parser.add_argument('--array', type=str, help='TP or TN array to generate xai for', required=True)
    parser.add_argument('--bias_label', type=int, help='0, 1, or None', default=None)
    args = parser.parse_args()
    exp_name = args.exp_name
    array_name = args.array
    bias_label = args.bias_label

    # exp_name = 'exp15E'
    # array_name = 'TN'
    # bias_label = None

    params = {'batch_size': 1,
              'array': array_name, #TP, TN, FP, FN
              'imagex': 173,
              'imagey': 211,
              'imagez': 155,
              'N_MAPS': 10, #number of subjects of each TP/TN/FP/FN array to generate saliency maps for
              'bias_label': bias_label, #0, 1, or None
              'smoothgrad': True, #True to generate smoothgrad maps
              'gradcam': False, #True to generate gradcam
              'conv_layer_name':'avgpool1',  #name of convolutional layer to visualize with gradcam
              'SMOOTH_SAMPLES': 20, #number of smoothgrad iterations
              'SMOOTH_NOISE': 0.2, #smoothgrad gaussian noise spread
              }

    model_name = 'best_model_' + exp_name + '.h5'

    home_dir = '/home/emma/Documents/SBB/'
    working_dir = home_dir + exp_name + '/'
    saliency_dir = working_dir + 'saliency/' + array_name + '/'  #for saliency output
    model_dir = working_dir + model_name

    #create directory for dataset
    Path(saliency_dir).mkdir(parents=True, exist_ok=True)

    #------------------- ------------------------------ ----------------------------#


    #retrieve dataframe with model predictions
    fn_eval = working_dir + 'preds_' + exp_name + '.csv'
    test = pd.read_csv(fn_eval, index_col = 'filename')


    #slice dataframe by desired bias label
    if params['bias_label'] is not None:
        test = test[test['bias_label']==params['bias_label']]


    test_ids=test.index.to_numpy()
    test_fpaths=test['filepath'].to_numpy()


    #convert true classes and preds to np arrays and get indices of TP/TN/FP/FN
    y_true = test['ground_truth'].to_numpy()
    y_pred = test['preds'].to_numpy()

    TP_idx = np.where((y_pred ==1) & (y_true==1))
    TN_idx = np.where((y_pred ==0) & (y_true==0))
    FP_idx = np.where((y_pred == 1) & (y_true == 0))
    FN_idx = np.where((y_pred == 0) & (y_true == 1))

    #arrays of subject IDs by classification
    TP_fpaths_full = test_fpaths[TP_idx]
    TN_fpaths_full = test_fpaths[TN_idx]
    FP_fpaths_full = test_fpaths[FP_idx]
    FN_fpaths_full = test_fpaths[FN_idx]
    FP_ids_full = test_ids[FP_idx]
    FN_ids_full = test_ids[FN_idx]
    TP_ids_full = test_ids[TP_idx]
    TN_ids_full = test_ids[TN_idx]

    TP_fpaths = subsample(TP_fpaths_full, params['N_MAPS'])
    TN_fpaths = subsample(TN_fpaths_full, params['N_MAPS'])
    # FP_fpaths = subsample(FP_fpaths_full, params['N_MAPS'])
    # FN_fpaths = subsample(FN_fpaths_full, params['N_MAPS'])
    # FP_ids = subsample(FP_ids_full, params['N_MAPS'])
    # FN_ids = subsample(FN_ids_full, params['N_MAPS'])
    TP_ids = subsample(TP_ids_full, params['N_MAPS'])
    TN_ids = subsample(TN_ids_full, params['N_MAPS'])


    X = np.zeros((params['N_MAPS'],params['imagex'], params['imagey'], params['imagez'], 1))

    if params['array']=='TP':
        data_generator=DataGenerator(TP_fpaths, params['N_MAPS'], (params['imagex'], params['imagey'], params['imagez']), False, fn_eval)
        X[:] = data_generator[0][0]
        array_IDs = TP_ids
        class_idx = 1



    if params['array']=='TN':
        data_generator=DataGenerator(TN_fpaths, params['N_MAPS'], (params['imagex'], params['imagey'], params['imagez']), False, fn_eval)
        X[:] = data_generator[0][0]
        array_IDs = TN_ids
        class_idx = 0


    # if params['array']=='FP':
    #     data_generator=DataGenerator(FP_fpaths, params['N_MAPS'], (params['imagex'], params['imagey'], params['imagez']), False, fn_eval)
    #     X[:] = data_generator[0][0]
    #     array_IDs = FP_ids
    #     class_idx = 1


    # if params['array']=='FN':
    #     data_generator=DataGenerator(FN_fpaths, params['N_MAPS'], (params['imagex'], params['imagey'], params['imagez']), False, fn_eval)
    #     X[:] = data_generator[0][0]
    #     array_IDs = FN_ids
    #     class_idx = 0


    #load model to generate saliency maps for
    model = tf.keras.models.load_model(model_dir)
    model.summary()



    if params['smoothgrad'] == True:

        for i, ID in enumerate(array_IDs):
            print('generating saliency map for {}'.format(ID))
            saliency_map = smoothgrad(X[i,:,:,:], model, class_idx)

            sitk_img = sitk.GetImageFromArray(np.swapaxes(saliency_map[:,:,:],0,2)) #save as x,y,z
            sitk.WriteImage(sitk_img, saliency_dir + 'SG_' + ID)


            # label = exp_name + ' | class label: ' + str(test.loc[ID, 'class_label']) + ' | bias label: ' + str(test.loc[ID, 'bias_label'])

            # #save figure of saliency overlaid on image
            # img = X[i,:]
            # img = (img - img.min()) / (img.max() - img.min() + K.epsilon())
            # img = img.reshape(img.shape[:-1])

            # plt.imshow(img[:,:,85], cmap=plt.cm.gray)
            # plt.imshow(saliency_map[0,:,:,85], cmap=plt.cm.viridis, alpha=.75)
            # plt.axis('off')
            # plt.title(label)
            # # plt.show()
            # plt.savefig(saliency_dir + 'SG_overlay_+' + ID[:-7] + '.png', bbox_inches='tight', dpi = 800)





    if params['gradcam'] == True:
        for i, ID in enumerate(array_IDs):
            print('generating GradCAM map for {}'.format(ID))
            cam = gradcam(X[i,:,:,:], model, params['conv_layer_name'], class_idx)

            sitk_img = sitk.GetImageFromArray(np.swapaxes(cam[:,:,:],0,2)) #save as x,y,z
            sitk.WriteImage(sitk_img, saliency_dir + 'GC_' + ID)

            # label = exp_name + ' | class label: ' + str(test.loc[ID, 'class_label']) + ' | bias label: ' + str(test.loc[ID, 'bias_label'])

            # #save figure of saliency overlaid on image
            # img = X[i,:]
            # img = (img - img.min()) / (img.max() - img.min() + K.epsilon())
            # img = img.reshape(img.shape[:-1])
            # plt.imshow(img, cmap=plt.cm.gray)
            # plt.imshow(cam, cmap=plt.cm.viridis, alpha=.75)
            # plt.axis('off')
            # plt.title(label)
            # # plt.show()
            # plt.savefig(saliency_dir + 'GC_overlay_+' + ID[:-7] + '.png', bbox_inches='tight', dpi = 800)

if __name__ == '__main__':
    main()
