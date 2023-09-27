#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emma

"""

import SimpleITK as sitk
import glob
import random
random.seed(6)
import argparse

def generate_avg_map(fpaths, bias_label):

    for i, fpath in enumerate(fpaths):
        print('reading image: {}'.format(i))
        img = sitk.ReadImage(fpath)
        img_arr = sitk.GetArrayFromImage(img)
        if(i == 0):
            summap = img_arr
        else:
            summap = img_arr+summap

    #divide by counter to average
    avg_arr = summap/(i+1)
    avg_img = sitk.GetImageFromArray(avg_arr)

    #Copy info to keep the orientation info
    avg_img.CopyInformation(img)

    #save average image
    sitk.WriteImage(avg_img, saliency_path + 'SG_avg_bias_' + str(bias_label) + '.nii.gz')

    return avg_arr, avg_img


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
parser.add_argument('--array', type=str, help='TP or TN array to generate xai for', required=True)
args = parser.parse_args()
expname = args.exp_name
arr = args.array


saliency_path = '/home/emma/Documents/SBB/' + expname + '/saliency/' + arr +'/'
n_maps = 10


flist = glob.glob(saliency_path + 'SG*.nii.gz')
b0_list = []
b1_list = []

for f in flist:
    if f.endswith('B.nii.gz'):
        b1_list.append(f)
    else:
        b0_list.append(f)


b0_list = random.sample(b0_list, n_maps)
b1_list = random.sample(b1_list, n_maps)


b1_arr, b1_img = generate_avg_map(b1_list,1)
b0_arr, b0_img = generate_avg_map(b0_list,0)
