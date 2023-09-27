#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:52:47 2023

@author: emma
"""

import ants
import numpy as np
import SimpleITK as sitk
import os
import random
import pandas as pd
import pickle
random.seed(6)

def load_image(fpath):
    # load image in both itk image type and ants image type

    itk_img = sitk.ReadImage(fpath) #sitk image (x,y,z)
    img_array = sitk.GetArrayFromImage(itk_img).astype(float) #numpy image (z,y,x)
    img_array = np.swapaxes(img_array,0,2) #x,y,z
    img = ants.from_numpy(img_array) #ants image

    # Set ants origin to same as itk img origin
    img.set_origin(itk_img.GetOrigin())

    return itk_img, img


def load_pca_model(pca_dir, model_name):
   f = open(pca_dir + model_name, 'rb')
   pca = pickle.load(f)
   f.close()
   return pca

def get_roi_bspline_params(df, region_val, label_atlas):
    '''
    get scattered bspline function parameters for a displacement field cropped to only contain information in a specific ROI

    ---inputs---
    df: random deformation field that you want to crop to ROI
    region val: label of ROI
    label_atlas: atlas with corresponding roi labels
    ---returns---
    parametric_data_roi: parametric data array for ROI to feed to bspline method
    scattered_data_roi: scattered data array for ROI to feed to bspline method
    '''
    mask = get_mask(label_atlas, region_val)

    #multiply disp field by mask
    #first need to collapse df vector image into each scalar component to multiply mask by,
    #then recombine into vector image with ComposeImageFilter
    #because MultiplyImageFilter won't take vector images as arguments
    np_df = sitk.GetArrayFromImage(df)
    df_x = np_df[:,:,:,0]
    df_y = np_df[:,:,:,1]
    df_z = np_df[:,:,:,2]

    np_mask = sitk.GetArrayFromImage(mask)
    np_x_mask = np.multiply(df_x, np_mask)
    np_y_mask = np.multiply(df_y, np_mask)
    np_z_mask = np.multiply(df_z, np_mask)

    # df_x_mask = sitk.GetImageFromArray(np_x_mask)
    # df_y_mask = sitk.GetImageFromArray(np_y_mask)
    # df_z_mask = sitk.GetImageFromArray(np_z_mask)

    # compose = sitk.ComposeImageFilter()
    # region_df = compose.Execute(df_x_mask, df_y_mask, df_z_mask)

    #apply cropped displacement to image with b spline
    # -> get all voxel coords and all disp field magnitudes in region to feed to b spline ?

    #use numpy array to get scattered data and parametric data

    #create parametric data array of size (# coordinates in roi, 3)
    n_coords = np.count_nonzero(np_mask)
    parametric_data_roi = np.zeros((n_coords, 3)) #3 columns correspond to x,y,z
    scattered_data_roi = np.zeros((n_coords, 3)) #3 columns correspond to x,y,z

    mask_nonzero = np.nonzero(np_mask) #tuple of ((z coordinates),(y coordinates),(x coordinates))
    #REMEMBER numpy array is in z,y,x, but we want sitk convention x,y,z
    parametric_data_roi[:,0]=mask_nonzero[2] # get sitk X (position 0) from np X (position 2)
    parametric_data_roi[:,1]=mask_nonzero[1] # get sitk Y (position 1) from np Y (position 1)
    parametric_data_roi[:,2]=mask_nonzero[0] # get sitk Z (position 2) from np Z (position 0)

    pd_int = parametric_data_roi.astype(int) #need coordinates as ints to access numpy array

    for i in range(n_coords):
        scattered_data_roi[i,0]=np_x_mask[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk X (position 0) from numpy (z,y,x) coordinates
        scattered_data_roi[i,1]=np_y_mask[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk Y (position 1) from numpy (z,y,x) coordinates
        scattered_data_roi[i,2]=np_z_mask[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk Z (position 2) from numpy (z,y,x) coordinates

    return parametric_data_roi, scattered_data_roi



def get_roi_bspline_params_fromPCA(df):
    '''
    get bspline parameters from a displacement field that's already cropped to a specific ROI
    ---inputs---
    df: ROI displacement field
    ---returns---
    parametric_data_roi: parametric data array for ROI to feed to bspline method
    scattered_data_roi: scattered data array for ROI to feed to bspline method
    '''

    #collapse df into x,y,z components
    np_df = sitk.GetArrayFromImage(df) #z,y,x,3
    df_x = np_df[:,:,:,0] #x vector component
    df_y = np_df[:,:,:,1] #y vector component
    df_z = np_df[:,:,:,2] #z vector component

    #create parametric data array of size (# coordinates in roi, 3)
    n_coords = max(np.count_nonzero(df_x), np.count_nonzero(df_y), np.count_nonzero(df_z))
    parametric_data_roi = np.zeros((n_coords, 3)) #3 columns correspond to x,y,z
    scattered_data_roi = np.zeros((n_coords, 3)) #3 columns correspond to x,y,z

    roi_nonzero = np.nonzero(np_df[:,:,:,0]) #tuple of ((z coordinates),(y coordinates),(x coordinates)) where there is a nonzero value
    #REMEMBER numpy array is in z,y,x, but we want sitk convention x,y,z
    parametric_data_roi[:,0]=roi_nonzero[2] # get sitk X (position 0) from np X (position 2)
    parametric_data_roi[:,1]=roi_nonzero[1] # get sitk Y (position 1) from np Y (position 1)
    parametric_data_roi[:,2]=roi_nonzero[0] # get sitk Z (position 2) from np Z (position 0)

    pd_int = parametric_data_roi.astype(int) #need coordinates as ints to access numpy array

    for i in range(n_coords):
        scattered_data_roi[i,0]=df_x[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk X (position 0) from numpy (z,y,x) coordinates
        scattered_data_roi[i,1]=df_y[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk Y (position 1) from numpy (z,y,x) coordinates
        scattered_data_roi[i,2]=df_z[pd_int[i][2], pd_int[i][1], pd_int[i][0]] #get sitk Z (position 2) from numpy (z,y,x) coordinates

    return parametric_data_roi, scattered_data_roi




def get_bspline_disp_field(ref_img, parametric_data, scattered_data,
                               meshSize=5, NFittingLevels=3):
    '''
    Generates a displacment field (sitk vector image type)

    ---inputs---
    ref_img: Ants image type
    parametric_data: location of points [x,y,z] corresponding to vectors in the same row from scattered_data
    scattered_data: [x,y,z] displacement vectors corresponding to locations in rows of parametric_data

    ---returns---
    inv_field: displacement field
    flag: if 1, means this displacement field transformation is not diffeomorphic

    '''
    bspline_ants = ants.fit_bspline_object_to_scattered_data(
        scattered_data, parametric_data,
        parametric_domain_origin=[0.0, 0.0, 0.0],
        parametric_domain_spacing=[1.0, 1.0, 1.0],
        parametric_domain_size = ref_img.shape,
        number_of_fitting_levels=NFittingLevels, mesh_size=meshSize)


    # convert bspline to numpy array
    bspline_arr = bspline_ants.numpy()

    #swap axes again
    bspline_arr = bspline_arr.swapaxes(0, 2)

    # convert numpy to sitk image
    bspline_itk = sitk.GetImageFromArray(bspline_arr, isVector=True)


    # invert displacement field generated from bspline
    inv_field = sitk.InvertDisplacementField(
            bspline_itk,
            maximumNumberOfIterations=20,
            maxErrorToleranceThreshold=0.01,
            meanErrorToleranceThreshold=0.0001,
            enforceBoundaryCondition=True)

    #check diffeomorphism
    flag, jd = diffeomorphic_check(inv_field)


    return inv_field, flag, jd




def diffeomorphic_check(disp_field):
    '''
    checks for negative values in jacobian determinant which indiciates non-diffeomorphic deformation
    returns 1 if non-diffeomorphic, 0 otherwise

    '''
    flag = 0
    jd = sitk.DisplacementFieldJacobianDeterminant(disp_field)
    jd_arr = sitk.GetArrayViewFromImage(jd)
    if jd_arr.min() < 0:
        flag = 1

    return flag, jd



def get_displ_list(n_samples):
    '''
    n_samples: number of samples in total dataset (D + ND)
    '''
    d = '/home/emma/Documents/SBB/IXI_nonlin_disp_fields'

    flist = [os.path.join(d, f) for f in os.listdir(d)]

    displ_list = random.sample(flist, n_samples)

    return displ_list


def get_mask(label_atlas, region_val):
    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetUpperThreshold(region_val)
    thresh_filter.SetLowerThreshold(region_val)
    thresh_filter.SetOutsideValue(0)
    thresh_filter.SetInsideValue(1)
    mask = thresh_filter.Execute(label_atlas)
    return mask



def pca_vector_to_itk(sample, mask_itk):
    '''
    convert velocity field array for local region (e.g. ROI) to sitk image

    ---inputs---
    sample: sampled ROI vector from trained PCA
    mask_itk: sitk image of ROI mask corresponding to PCA model (required for correctly reshaping sample)

    ---returns---
    sample_itk: sitk image of ROI sampled from PCA model
    '''
    #flatten the ROI mask
    mask_np=sitk.GetArrayFromImage(mask_itk)
    mask_displ=np.repeat(mask_np[:,:,:,np.newaxis],3,axis=3)
    mask_flattended=np.ravel(mask_displ,order='F')

    #reshape sampled component into vector of full size image
    sample_flattened = np.zeros_like(mask_flattended).astype('float64')
    sample_arr = np.squeeze(np.asarray(sample)) #convert sample to insert from matrix to array
    loc = np.nonzero(mask_flattended)[0] #locations to insert
    sample_flattened[loc] = sample_arr

    #reshape vector to 3d array
    sample_3d = np.reshape(sample_flattened, mask_displ.shape, order='F')

    #get sitk image from numpy array
    sample_itk = sitk.GetImageFromArray(sample_3d, isVector=True)

    return sample_itk


def pca_vector_to_itk_full(sample, ref_img):
    '''
    convert velocity field array for full image (e.g. subject variability) to sitk image

    ---inputs---
    sample: sampled vector from trained PCA (full size image)
    ref_img: reference image to reshape vector to the same size
    ---returns---
    sample_itk: sitk image of ROI sampled from PCA model
    '''
    #flatten the ref image
    ref_np=sitk.GetArrayFromImage(ref_img)
    ref_displ=np.repeat(ref_np[:,:,:,np.newaxis],3,axis=3)

    sample_arr = np.squeeze(np.asarray(sample)) #convert sample to insert from matrix to array

    #reshape vector to 3d array
    sample_3d = np.reshape(sample_arr, ref_displ.shape, order='F')

    #get sitk image from numpy array
    sample_itk = sitk.GetImageFromArray(sample_3d, isVector=True)

    return sample_itk


def svf_scaling_and_squaring(velo_field_itk, accuracy=4, compute_inverse=False):
    '''
    compute displacement field from velocity field
    ref: Arsigny, V., Commowick, O., Pennec, X., Ayache, N.: A Log-Euclidean Framework for Statistics on Diffeomorphisms. In: Larsen, R., Nielsen, M., Sporring, J.
    (eds.) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2006. pp. 924–931. Lecture Notes in Computer Science, Springer, Berlin, Heidelberg (2006)
    '''
    #scale field according to the desired integration accuracy/steps (2^steps)
    velo_field_np=np.float32(sitk.GetArrayFromImage(-velo_field_itk if compute_inverse else velo_field_itk)/(2**accuracy))
    velo=sitk.GetImageFromArray(velo_field_np,isVector=True)
    velo.CopyInformation(velo_field_itk)

    #prepare warper for squaring step (velo_n+1 = velo_n \circ velo_n)
    warper=sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkLinear)
    warper.SetOutputParameteresFromImage(velo_field_itk)

    #squaring step
    for i in range(accuracy):
        temp=warper.Execute(velo,velo)
        velo=velo+temp

    return velo


def transport_displ_field(f_velo_itk,g_velo_itk):
    '''
    conjugate action mechanism to transport dense velocity field to "subject" space
    ref: Lorenzi, M., Pennec, X.: Geodesics, Parallel Transport & One-Parameter Subgroups for Diffeomorphic Image Registration. International Journal of Computer
    Vision 105(2), 111–127 (2013)
    '''
    #computes e=exp(-f)\circ exp(g) \circ exp(f)
    f_displ_itk=svf_scaling_and_squaring(f_velo_itk)
    f_displ_inv_itk=svf_scaling_and_squaring(f_velo_itk,compute_inverse=True)
    g_displ_itk=svf_scaling_and_squaring(g_velo_itk, compute_inverse=True)

    warper=sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkLinear)
    warper.SetOutputParameteresFromImage(f_displ_itk)
    temp_1=warper.Execute(g_displ_itk,f_displ_itk)
    temp_2=warper.Execute(f_displ_inv_itk,temp_1+f_displ_itk)

    return temp_2+temp_1+f_displ_itk


def merge_shuffle_data(df_list):
    '''
    input -> list of dataframes to combine
    output -> dataframe with all data combined and randomly shuffled
    '''
    full_df = pd.concat(df_list)
    full_df = full_df.sample(frac=1, random_state=42)

    return full_df

def write_flag(fpath, name, n, idx, flagtype):
    '''
    ---inputs---
    fpath: filepath to csv file with flags
    name: experiment name
    n: sample number
    idx: index in df to write flags about this sample to
    flagtype: name of column corresponding to this flag
    '''
    flag_df = pd.read_csv(fpath, index_col='Index')
    flag_df.loc[idx, 'Experiment Name'] = name
    flag_df.loc[idx, 'Sample Number'] = str(n).zfill(5)
    flag_df.loc[idx, flagtype] = 1
    flag_df.to_csv(fpath)

    return


def bspl_params_from_df(df, label, n_samples):
    '''
    get x,y,z disp_arr and parametric_data from dataframe containing parameters for generating dataset

    ---inputs---
    df: dataset parameter pandas dataframe
    label: "bias" or "disease", depending on which feature you want the bspline parameters for

    --returns---
    parametric_data: location of points [x,y,z] corresponding to vectors in the same row from scattered_data
                     number of rows = number of locations where disp vectors are defined, 3 columns correspond to x,y,z locations
    x_disp_arr: x displacements -> rows correspond to desired displacements at corresponding locations in parametric_data, columns correspond to number of samples in dataset
    y_disp_arr: y displacements -> rows correspond to desired displacements at corresponding locations in parametric_data, columns correspond to number of samples in dataset
    z_disp_arr: z displacements -> rows correspond to desired displacements at corresponding locations in parametric_data, columns correspond to number of samples in dataset

    '''
    if label == 'bias':
        ft = 'B'
    elif label == 'disease':
        ft = 'D'
    else:
        print('incorrect label entered - must be "bias" or "disease"')

    #get locations, displacements, and standard deviations from dataframe
    x_loc = df[ft+'_x_loc'].dropna().to_list()
    y_loc = df[ft+'_y_loc'].dropna().to_list()
    z_loc = df[ft+'_z_loc'].dropna().to_list()

    x_disp = df[ft+'_x_disp'].dropna().to_list()
    y_disp = df[ft+'_y_disp'].dropna().to_list()
    z_disp = df[ft+'_z_disp'].dropna().to_list()

    x_sd = df[ft+'_x_sd'].dropna().to_list()
    y_sd = df[ft+'_y_sd'].dropna().to_list()
    z_sd = df[ft+'_z_sd'].dropna().to_list()

    #define parametric data -> location of points [x,y,z] corresponding to vectors in the same row from scattered_data
    parametric_data = np.zeros((max(len(x_loc), len(y_loc), len(z_loc)), 3))
    parametric_data[:,0] = x_loc # x column
    parametric_data[:,1] = y_loc # y column
    parametric_data[:,2] = z_loc # z column

    #generate N samples from normal distributions based on desired x and y displacements
    x_disp_arr = np.zeros((len(x_loc), n_samples))
    y_disp_arr = np.zeros((len(y_loc), n_samples))
    z_disp_arr = np.zeros((len(z_loc), n_samples))

    for i,x in enumerate(x_disp):
        x_disp_arr[i,:] = np.random.normal(x, x_sd[i], n_samples)

    for i,y in enumerate(y_disp):
        y_disp_arr[i,:] = np.random.normal(y, y_sd[i], n_samples)

    for i,z in enumerate(z_disp):
        z_disp_arr[i,:] = np.random.normal(z, z_sd[i], n_samples)

    return parametric_data, x_disp_arr, y_disp_arr, z_disp_arr
