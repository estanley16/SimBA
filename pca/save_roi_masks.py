#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:34:13 2023

@author: emma
"""

import numpy as np 
import pandas as pd
import SimpleITK as sitk
import os

def get_mask(label_atlas, region_val):
    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetUpperThreshold(region_val)
    thresh_filter.SetLowerThreshold(region_val)
    thresh_filter.SetOutsideValue(0)
    thresh_filter.SetInsideValue(1)
    mask = thresh_filter.Execute(label_atlas)
    return mask


lpath = '/home/emma/Documents/SBB/atlas/sri24_spgr_lpba40_labels_RAI_ROI.nii.gz' 
label_atlas = sitk.ReadImage(lpath)

dfpath = '/home/emma/Documents/SBB/atlas/LPBA40-labels.csv'
df = pd.read_csv(dfpath)

savepath = '/home/emma/Documents/SBB/atlas/roi_masks/'
os.makedirs(savepath, exist_ok=True)

region_vals = df['0'].tolist()

for i,val in enumerate(region_vals): 
    mask = get_mask(label_atlas, val)
    label = df.loc[i, 'background'] + '_mask_' + str(val)
    sitk.WriteImage(mask, savepath + label + '.nii.gz')
