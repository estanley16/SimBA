#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May 31 14:48:35 2023

@author: emma

from generate_bspline_dataset_3d_ROI_v10.py
*modifying to introduce combination of intensity + morph bias


"""

import ants
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import pandas as pd
import utils
from numpy.random import Generator, PCG64
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split

# ----------------------------------------
#                arguments
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--d_roi', type=int, help='label of disease ROI', required=True)
parser.add_argument('--b_roi', type=int, help='label of bias ROI')
parser.add_argument('--isv', type=int, help='1 to add intersubject variability with real displacement fields, else 0', required=True)
parser.add_argument('--split', type=str, help='split to generate data for')
parser.add_argument('--add_bias', type=int, help='whether or not to add bias effect, 0 = no, 1 = yes')
parser.add_argument('--expname', type=str, required=True)
args = parser.parse_args()


# ----------------------------------------
#                  setup
# ----------------------------------------
main_dir = '/home/emma/Documents/SBB/datasets/'



save_dir = main_dir + args.expname + '/' + args.split + '/'
#create directory for dataset
Path(save_dir).mkdir(parents=True, exist_ok=True)

# read template image (3d)
fpath = '/home/emma/Documents/SBB/atlas/sri24_spgr_RAI_ROI.nii.gz' #atlas with RAI orientation + origin set to 0,0,0 to be consistent with itk + bspline method
itk_img, img = utils.load_image(fpath)

# read label atlas
lpath = '/home/emma/Documents/SBB/atlas/sri24_spgr_lpba40_labels_RAI_ROI.nii.gz' #path to label atlas
label_atlas = sitk.ReadImage(lpath)

#create itk objects for later on
writer = sitk.ImageFileWriter()
warper = sitk.WarpImageFilter()
warper.SetInterpolator(sitk.sitkLinear)
warper.SetOutputParameteresFromImage(itk_img)



# load correct distribution based on split
dataframe = pd.read_csv('/home/emma/Documents/SBB/' + args.expname + '/effect_distributions/' + args.split + '_dst.csv', index_col=0)

#sort dataframe so that images with the morph effect are added first !
custom_order = [0, 2, 1, 3] # Define custom sorting order

# Sort the DataFrame based on the custom order of values in 'Column_Name'
dataframe = dataframe[dataframe['bias_label'].isin(custom_order)].sort_values(by='bias_label', key=lambda x: x.map({v: i for i, v in enumerate(custom_order)}))


# extract effect and isv distribution values
effect_dst = dataframe['effect_dst'].values
isv_dst = dataframe['isv_dst'].values


# extract bias labels
bias_labels = dataframe['bias_label'].values


n_samples = len(dataframe)


SEED=len(dataframe) #seed for generating bias effect distribution (want it to be different between splits)
numpy_randomGen = Generator(PCG64(SEED))

# ----------------------------------------
#           effect models (PCA)
# ----------------------------------------
# load required PCA models and define sampling distribution along components
pca_dir = '/home/emma/Documents/SBB/pca_models_velo_50/' #path to pca model directory


# ------ inter-subject variability effects ------
if args.isv == 1:
    pca_isv = utils.load_pca_model(pca_dir, 'sri24_spgr_RAI_lpba40_isv') #open PCA model for subject effects
    n_comps_isv = 10 #number of PCA components to sample for subject effects


# ------ disease effect ------
pca_d = utils.load_pca_model(pca_dir, 'sri24_spgr_RAI_lpba40_' + str(args.d_roi)) #open PCA model for disease effect

# ------ bias effects ------

#intensity effect
def change_contrast(img, factor, min_range, max_range):
    half_range = (max_range-min_range)/2
    new_img = np.clip(factor*(img - half_range)+half_range, min_range, max_range)
    scaled_img = (new_img - min_range) / (max_range - min_range)
    return scaled_img


#define dictionary for intensity values
#key = bias label, value = factor to change contrast by
contrast_dict = {
                0: 1.2, #B + I
                1: 1.2, #NB + I
                2: 1, #B + NI
                3: 1 #NB + NI
                }

#morphological effect
pca_b = utils.load_pca_model(pca_dir, 'sri24_spgr_RAI_lpba40_' + str(args.b_roi)) #open PCA model for bias effect
num_b = len(dataframe.loc[dataframe['bias_label']==0]) + len(dataframe.loc[dataframe['bias_label']==2]) # number of images that will contain the bias feature

#define B/NB sampling distribution
bias_dst_mean = 2 #mean value of component for generating distribution of samples
bias_dst_sd = 1 #standard deviation for generating distirbution of samples
bias_dst = numpy_randomGen.normal(bias_dst_mean, bias_dst_sd, num_b) #get samples from bias effect distribution



# ----------------------------------------
#           data generation
# ----------------------------------------

fname_lst = []
fpath_lst = []

print('generating data')
for n in range(n_samples):
    print(n)
 
    #get label for image
    label = str(np.around(isv_dst[n],3)) + '_S_' + str(np.around(effect_dst[n],3)) + '_D'
    label = label + '_'+ str(bias_labels[n]) +'_B'

    # ---------- add data information to dataframe columns ------------
    dataframe.loc[dataframe.index[n], 'filepath'] = save_dir + str(n).zfill(5) + '_' + label + '.nii.gz'
    dataframe.loc[dataframe.index[n], 'filename'] = str(n).zfill(5) + '_' + label + '.nii.gz'
    fname_lst.append(str(n).zfill(5) + '_' + label + '.nii.gz')
    fpath_lst.append(save_dir + str(n).zfill(5) + '_' + label + '.nii.gz')
    
    # ------- disease effects -------
    # sample (forward) velocity field from first component in disease effect pca model
    print('d sampling val: {}'.format(effect_dst[n]))
    vf_d_np = pca_d.translation_vector + pca_d.basis[:,0]*np.sqrt(pca_d.eigenvalues[0])*effect_dst[n] #disease velocity field as numpy arr
    vf_d = utils.pca_vector_to_itk(vf_d_np, utils.get_mask(label_atlas, args.d_roi)) #disease velocity field as sitk image type

    #get parametric and scattered data arrays for disease (forward) velocity field
    parametric_data_roi, scattered_data_roi = utils.get_roi_bspline_params_fromPCA(vf_d)


    # ------- morph bias effects -------
    if args.add_bias == 1 and (bias_labels[n] == 0 or bias_labels[n] == 2): #groups 0 and 2 have the morph bias effect
          #sample (fwd) velocity field from from first component in bias effect pca model
          print('b sampling val: {}'.format(bias_dst[n]))
          vf_b_np = pca_b.translation_vector + pca_b.basis[:,0]*np.sqrt(pca_b.eigenvalues[0])*bias_dst[n] #bias velocity field as np arr
          vf_b = utils.pca_vector_to_itk(vf_b_np, utils.get_mask(label_atlas, args.b_roi)) #bias velocity field as sitk image type
    
          #get parametric and scattered data arrays for (fwd) disease velocity field
          parametric_data_roi_b, scattered_data_roi_b = utils.get_roi_bspline_params_fromPCA(vf_b)
    
          #concatenate disease and bias arrays
          parametric_data_roi = np.vstack((parametric_data_roi, parametric_data_roi_b))
          scattered_data_roi = np.vstack((scattered_data_roi, scattered_data_roi_b))


    # ------- generate dense displacement field for disease + bias effect -------
    vf, _, jd = utils.get_bspline_disp_field(img, parametric_data_roi, scattered_data_roi, meshSize=10)

    #perform scaling and squaring on velocity field to get displacement field for the disease + bias effect
    #get the inverse field since that's what we want to apply to the image
    df = utils.svf_scaling_and_squaring(vf, compute_inverse=True)



    # ------- include inter-subject variability effects -------
    if args.isv == 1:
        #sample (fwd) velocity field from first n components isv effect pca mode
        print('isv sampling val: {}'.format(isv_dst[n]))
        vf_isv_np = pca_isv.translation_vector + pca_isv.basis[:,:n_comps_isv]*np.sqrt(pca_isv.eigenvalues[:n_comps_isv])*isv_dst[n] #subject velocity field as np arr
        vf_isv = utils.pca_vector_to_itk_full(vf_isv_np, itk_img) #subject velocity field as sitk image type

        #perform scaling and squaring on velocity field to get displacement field for the subject effect
        #need inverse (atlas -> subject) disp fields for generating "subject" template
        inv_df_isv = utils.svf_scaling_and_squaring(vf_isv, compute_inverse=True)


        #generate subject image
        s_img = warper.Execute(itk_img, inv_df_isv)

        #generate combined df for isv + effect (conjugate action mechanism)
        trans_effect_df=utils.transport_displ_field(vf_isv, vf)

        #apply transformed effect to subject
        gen_img=warper.Execute(s_img, trans_effect_df)


    # ------- without inter-subject variability effects -------
    else:
        #apply primary deformation field to image
        gen_img = warper.Execute(itk_img, df)

    # --------- add intensity bias effect -> convert to numpy array, change contrast, convert back to sitk
    if args.add_bias == 1:
        gen_img_np = sitk.GetArrayFromImage(gen_img)
        gen_img_np_cont = change_contrast(gen_img_np, contrast_dict.get(bias_labels[n]), gen_img_np.min(), gen_img_np.max())
        gen_img = sitk.GetImageFromArray(gen_img_np_cont)
    
    
    # ------- save generated image -------
    print('saving sample {}'.format(n))
    writer.SetFileName(save_dir + str(n).zfill(5) + '_' + label + '.nii.gz')
    writer.Execute(gen_img)


# ---------------- save dataframe ---------------------

dataframe['filepath']=fpath_lst
dataframe['filename']=fname_lst
dataframe.to_csv('/home/emma/Documents/SBB/' + args.expname + '/' + args.split + '.csv')
