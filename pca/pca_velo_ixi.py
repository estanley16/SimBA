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




# bspl_warped = warper.Execute(atlas_itk, df)
# sitk.WriteImage(bspl_warped,'/home/emma/Documents/SBB/sandbox/pca_Lhippocampus_b0_test_atlas_bspl.nii.gz')
#####################################################################################
