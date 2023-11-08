'''
author: @emma 

compute importance scores for each brain ROI based on a saliency map
importance score = (#salient voxels in region)/(#total voxels in region) x 100%

                    
JMI paper code modified to SimBA data 2023/07/04
'''
import os
import vtk
import argparse
import pandas as pd
import numpy as np

def makeParser():
    parser = argparse.ArgumentParser(description='Selecting which dataset to visualize')
    parser.add_argument('--smap', type=str, metavar='PATH', help='Path to cleaned (lower thresholded) registered avg saliency map nifti file', required=True)
    parser.add_argument('--expname', type=str, help='experiment name (name of folder with other experiment data)', required=True)
    parser.add_argument('--group', type=str, help='TP or TN', required=True)
    parser.add_argument('--bias_label', type=str, help='0 or 1')
    return parser


def nifti_reader(fname):

    #reader: NIFTI
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader



def global_threshold_segment(imagedata, thresh_lower, thresh_upper):
    '''
    apply global thresholding to assign value of 1 to segmented regions
    (for use when calculating overlap measures)

    inputs:
    -image data: vtkImageData type
    -thresh_lower: lower threshold for segmentation
    -thresh_upper: upper threshold for segmentation

    returns: thresholded image of vtkImageData type
    '''
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(imagedata)
    threshold.ThresholdBetween(thresh_lower, thresh_upper) #thresholds between, inclusive, double
    threshold.ReplaceInOn()
    threshold.SetInValue(1) #set values between lower and upper threshold to 1
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0) #set values out of threshold range to 0
    threshold.Update()
    return threshold.GetOutput()

def cast_to_double(image):
    '''
    cast vtkImageData scalar type to double
    '''
    cast = vtk.vtkImageCast()
    cast.SetOutputScalarTypeToDouble()
    cast.SetInputData(image)
    cast.Update()
    return cast.GetOutput()


def multiply_images(image1, image2):
    '''
    inputs: two images to multiply, vtkImageData
    outputs: resulting image, vtkImageData
    '''
    #cast both images to scalar type = double
    image1 = cast_to_double(image1)
    image2 = cast_to_double(image2)

    math = vtk.vtkImageMathematics()
    math.SetOperationToMultiply()
    math.SetInput1Data(image1)
    math.SetInput2Data(image2)
    math.Update()
    return math.GetOutput()


def voxel_count(image):
    '''
    get amount of nonzero voxels in a vtk image
    input: image to count non-zero voxels for, vtkImageData
    output: number of nonzero voxels in image
    '''
    stat = vtk.vtkImageAccumulate()
    stat.SetInputData(image)
    stat.IgnoreZeroOn()
    stat.Update()
    count = stat.GetVoxelCount()
    return count

def voxel_mean(image):
    '''
    get mean of nonzero voxels in a vtk image
    input: image to determine mean of non-zero voxels for, vtkImageData
    output: mean of nonzero voxels in image
    '''
    stat = vtk.vtkImageAccumulate()
    stat.SetInputData(image)
    stat.IgnoreZeroOn()
    stat.Update()
    mean = stat.GetMean()
    return mean


def main():
    #parse arguments
    parser = makeParser()
    args = parser.parse_args()

    WORKING_DIR = '/home/emma/Documents/SBB/'
    ATLAS_PATH = WORKING_DIR + 'atlas/sri24_spgr_lpba40_labels_RAI_ROI.nii.gz' #path to LPBA40 parcellation atlas
    ATLAS_INFO_PATH = WORKING_DIR +'atlas/LPBA40-labels.csv'

    #read parcellation atlas (transformed to NIHPD space)
    atlas_reader = nifti_reader(ATLAS_PATH)
    atlas = atlas_reader.GetOutput()

    #read saliency map
    smap_reader = nifti_reader(args.smap)
    smap = smap_reader.GetOutput()

    #create dataframe to store saliency map information corresponding to region and label
    #re-format the csv provided with the parcellation atlas
    df = pd.read_csv(ATLAS_INFO_PATH, usecols=('0', 'background'), index_col = '0')
    df.index.names = ['Label']

    region_vals = df.index.tolist() #list of all values corresponding to regions from the parcellation atlas

    for val in region_vals:

        THRESH = val

        #get image with only ROI corresponding to val set to 1
        ROI = global_threshold_segment(atlas, THRESH, THRESH)

        #multiply saliency map * ROI
        smap_ROI = multiply_images(ROI, smap)

        #get total number of voxels in ROI and number of salient voxels in ROI
        n_ROI = voxel_count(ROI)
        n_smap_ROI = voxel_count(smap_ROI)

        mean_smap = voxel_mean(smap_ROI)[0]

        #compute saliency score for that ROI = salient voxels in region/total voxels in region x 100%
        score = n_smap_ROI/n_ROI*100

        # add score to corresponding cell in dataframe
        df.at[val, 'smap_score (%)']=score
        df.at[val, 'smap_region_mean']=mean_smap

    #normalized mean
    df['smap_region_mean'] = df['smap_region_mean'].replace({0:np.nan})
    df['normalized_smap_region_mean'] = (df['smap_region_mean']-df['smap_region_mean'].min())/(df['smap_region_mean'].max()-df['smap_region_mean'].min())

    #weighted saliency score
    df['weighted_smap_score (%)'] = df['normalized_smap_region_mean'] * df['smap_score (%)']

    #save csv file
    df.to_csv(WORKING_DIR + args.expname + '/saliency/' + args.group + '/' + args.expname + '_' + args.group + '_biaslabel' + args.bias_label + '_avg_ROI_saliency.csv')


if __name__ == '__main__':
    main()
