#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:37:41 2023

@author: emma
"""

# python imports
import os
import glob
import sys
import random
import tensorflow as tf
import numpy as np

os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random

SEED=8
tf.random.set_seed(8)
np.random.seed(8)
random.seed(8)

# import scipy.io as sio
# import matplotlib.pyplot as plt
# import pca_utils as utils
import subspacemodels
# import kernels
# import dists
# import prdc
import pickle
import SimpleITK as sitk
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--velo_path', type=str, help='full directory path containing velocity fields for PCA', required=True)
# parser.add_argument('--mask_file', type=str, help='full file path to desired ROI mask (can also be the full brain mask)', required=True)
# parser.add_argument('--savename', type=str, help='name for saved PCA model, i.e., region value from LPBA40 atlas, or "isv"')
parser.add_argument('--region', type=int, help='LPBA40 region value')
parser.add_argument('--var_ret', type=float, help='variability retained in PCA model', default=1)
args = parser.parse_args()

#load region mask
mask_dir = '/home/emma/Documents/SBB/atlas/roi_masks/'
# mask_file=args.mask_file #ROI mask 
mask_file = glob.glob(mask_dir + "*" + str(args.region) + ".nii.gz")[0]

mask_itk=sitk.ReadImage(mask_file)
mask_np=sitk.GetArrayFromImage(mask_itk)
mask_velo=np.repeat(mask_np[:,:,:,np.newaxis],3,axis=3) #match size of velocity field (3 directions)
mask_flattened=np.ravel(mask_velo,order='F')

# atlas_file='/home/emma/Documents/SBB/atlas/sri24_spgr_RAI_ROI.nii.gz'
# atlas_itk=sitk.ReadImage(atlas_file)
# atlas_np=sitk.GetArrayFromImage(atlas_itk)


#load velocity field files
# velo_dir='/home/emma/Documents/SBB/IXI_nonlin_disp_fields_reduced'
velo_dir=args.velo_path
velo_files = sorted(glob.glob(os.path.join(velo_dir, '*velo.nii.gz')))
   
#directory for saving pca models 
save_dir = '/home/emma/Documents/SBB/pca_models_velo_50/'

data_matrix_velo=None

#read velo field files, multiply with ROI mask, and add to numpy array 
for f in velo_files:        
    print(f)
    
    velo_itk=sitk.ReadImage(f)
    velo_np=sitk.GetArrayFromImage(velo_itk)

    if data_matrix_velo is None:
        data_matrix_velo=np.matrix(np.ravel(velo_np,order='F')).T[mask_flattened==1,:]
        print(str(data_matrix_velo.shape))
    else:
        data_matrix_velo=np.append(data_matrix_velo,np.matrix(np.ravel(velo_np,order='F')).T[mask_flattened==1,:],axis=1)
        print(str(data_matrix_velo.shape))


#generate PCA model 
variability_retained=args.var_ret
pca_model=subspacemodels.SubspaceModelGenerator.compute_pca_subspace(data_matrix_velo,variability_retained)

#save pca model 
f = open(save_dir + 'sri24_spgr_RAI_lpba40_' + str(args.region), 'wb')
pickle.dump(pca_model, f)
f.close()


#####################################################################################
# #test -> load pca model
# f = open(save_dir + 'L_hippocampus_test', 'rb')
# pca_loaded = pickle.load(f)
# f.close()
# b = 0 #sampling scalar 
# sample_loaded = pca_loaded.translation_vector + pca_loaded.basis[:,0] * np.sqrt(pca_loaded.eigenvalues[0]) * b


#####################################################################################
# #test -> get 1st component eigenvector, sample from it, reshape into full size image

# b = 0 #sampling scalar 
# sample = pca_model.translation_vector + pca_model.basis[:,0] * np.sqrt(pca_model.eigenvalues[0]) * b

# #reshape sampled component into vector of full size image 
# sample_flattened = np.zeros_like(mask_flattened).astype('float64')
# sample_arr = np.squeeze(np.asarray(sample)) #convert sample to insert from matrix to array
# loc = np.nonzero(mask_flattened)[0] #locations to insert 
# sample_flattened[loc] = sample_arr 

# #reshape vector to 3d array
# sample_3d = np.reshape(sample_flattened, mask_velo.shape, order='F')

# #get sitk image from numpy array 
# sample_itk = sitk.GetImageFromArray(sample_3d, isVector=True)
# sitk.WriteImage(sample_itk,'/home/emma/Documents/SBB/sandbox/pca_Lhippocampus_b0_test.nii.gz')



#####################################################################################
# # test -> apply df to atlas
# warper = sitk.WarpImageFilter()
# warper.SetInterpolator(sitk.sitkLinear)
# warper.SetOutputParameteresFromImage(atlas_itk)

# # warped = warper.Execute(atlas_itk, sample_itk)
# # sitk.WriteImage(warped,'/home/emma/Documents/SBB/sandbox/pca_Lhippocampus_b0_test_atlas.nii.gz')

# from disp_field_utils import get_bspline_disp_field, get_velo_list, diffeomorphic_check, get_roi_bspline_params, composite_transform
# import ants

# lpath = '/home/emma/Documents/SBB/atlas/sri24_spgr_lpba40_labels_RAI_ROI.nii.gz' 
# label = sitk.ReadImage(lpath)

# img_array = sitk.GetArrayFromImage(atlas_itk).astype(float) #numpy image (z,y,x)
# img_array = np.swapaxes(img_array,0,2) #x,y,z
# img = ants.from_numpy(img_array) #ants image


# inv_df_d = sitk.InvertDisplacementField(
#                 sample_itk,
#                 maximumNumberOfIterations=20,
#                 maxErrorToleranceThreshold=0.01,
#                 meanErrorToleranceThreshold=0.0001,
#                 enforceBoundaryCondition=True)

            
# #get parametric and scattered data arrays for disease disp field
# parametric_data_roi, scattered_data_roi = get_roi_bspline_params(inv_df_d, 166, label) #THIS FXN COULD BE MODIFIED LATER SINCE DF SHOULD ALREADY BE MASKED / ONLY HAVE INFO FROM ROI
# df, flag = get_bspline_disp_field(img, parametric_data_roi, scattered_data_roi, meshSize=20)

# bspl_warped = warper.Execute(atlas_itk, df)
# sitk.WriteImage(bspl_warped,'/home/emma/Documents/SBB/sandbox/pca_Lhippocampus_b0_test_atlas_bspl.nii.gz')
#####################################################################################
