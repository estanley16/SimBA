#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:28:54 2023

@author: emma
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from numpy.random import Generator, PCG64
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from utils import merge_shuffle_data
import seaborn as sns
import matplotlib.pyplot as plt


# --------- parameters + setup ---------------
SEED_D  = 7 #seed for disease class
SEED_ND = 4 #seed for non-disease class
SEED_S = 2 #seed for subject effects
TOTAL_N = 6000 #total number of samples to start with
D_FRAC = 0.5 #fraction of total samples in disease class
ISV = True
MU_D = 1 #mean of distribution for disease class
MU_ND = -1 #mean of distribution for non-disease class
SD_D = 1 #std of effect sampling dst
MU_S = 0 #mean of subject sampling dst
SD_S = 1 #std of subject sampling dst
LOWER_S = -3.5 #lower bound of subject sampling dst
UPPER_S = 3.5 #upper bound of subject sampling dst
BINS_D = 10 #number of bins for disease effect stratification
BINS_S = 10 #number of bins for subject effect stratification
TRAIN_FRAC = 0.5 #fraction of dataset to split into training set
VAL_FRAC = 0.25 #fraction of dataset to split into validation set
TEST_FRAC = 0.25 #fraction of dataset to split into test set
EXP = 'test_exp' #experiment name

#create directory to save distribution values to
main_dir = '/home/emma/Documents/SBB/'
Path(main_dir + EXP + '/effect_distributions').mkdir(parents=True, exist_ok=True)
save_dir = main_dir + EXP + '/effect_distributions/'

#%%
#function to generate sampling distributions
def get_subject_distributions(seed, mu, sd, num, d_frac, lower_bound, upper_bound, bins):
    '''
    get dataframe with subject effects for each disease class
    
    -----inputs------
    -seed: seed for deterministic random number generation
    -mu: mean value of subject effect sampling distribution
    -sd: standard deviation of subject effect sampling distribution
    -num: number of samples to generate
    -d_frac: fraction of all data defined as the disease class
    -lower_bound: lower bound of truncated gaussian distribution for subject sampling
    -upper_bound: upper bound of truncated gaussian distribution for subject sampling
    -bins: number of bins for subject effect stratification
    -----outputs-----
    -isv_nd: subject effect distribution for the non-disease class
    -isd_d: subject effect distribution for the disease_class
    '''
    #define subject effect distribution
    numpy_randomGen = Generator(PCG64(seed))
    truncnorm.random_state=numpy_randomGen

    isv_dst_mean = mu #mean value of component for generating distribution of samples
    isv_dst_sd = sd #standard deviation for generating distirbution of samples
    isv_dst_bounds = [lower_bound, upper_bound]
    isv_dst_raw = truncnorm.rvs((isv_dst_bounds[0]-isv_dst_mean)/isv_dst_sd, (isv_dst_bounds[1]-isv_dst_mean)/isv_dst_sd, loc=isv_dst_mean, scale=isv_dst_sd, size=num)
    isv_bins = pd.cut(isv_dst_raw, bins=bins, labels=False)
    s = {'isv_dst': isv_dst_raw,
         'isv_bin': isv_bins}

    isv_df = pd.DataFrame(data=s)
    w = isv_df.isv_bin.value_counts()
    bins_to_remove = w.index[w<=1].tolist()
    print('dropping {} bins from class sampling step'.format(len(bins_to_remove)))
    isv_to_split = isv_df[~isv_df['isv_bin'].isin(bins_to_remove)]
    isv_nd, isv_d = train_test_split(isv_to_split, test_size = d_frac, random_state=42, stratify=isv_to_split[['isv_bin']])

    return isv_nd, isv_d

def get_class_distributions(seed, mu, sd, num_samples, isv_dst, isv_bins, bins_d, bins_s, class_label):
    '''
    get dataframe with disease + subject effects for all subjects in a given class
    
    -----inputs------
    -seed: seed for deterministic random number generation
    -mu: mean value of disease effect sampling distribution
    -sd: standard deviation of disease effect sampling distribution
    -num_samples: number of samples to generate for this class
    -isv_dst: stratified subject effect distribution for this class
    -isv_bins: bin values from subject effect stratification step
    -bins_d: number of stratificatin bins for disease effects 
    -bins_s: number of stratification bins defined for subject effects
    -class_label: numeric label for this disease class
    -----outputs-----
    -df: dataframe with subject effects, disease effects, bias labels, and class labels for this disease class
    '''    
    numpy_randomGen = Generator(PCG64(seed))
    
    #generate disease effect distribution
    effect_dst_mean = mu #mean value of component for generating distribution of samples
    effect_dst_sd = sd #standard deviation for generating distribution of samples
    effect_dst_raw = numpy_randomGen.normal(effect_dst_mean, effect_dst_sd, num_samples) #get samples from disease effect distribution
    
    effect_bins = pd.cut(effect_dst_raw, bins=bins_d, labels=False)
    
    d = {'isv_dst': isv_dst,
         'effect_dst': effect_dst_raw,
         'isv_bin': isv_bins,
         'effect_bin': effect_bins}
    
    df = pd.DataFrame(data=d)
    df['both_bins'] = df['isv_bin'].astype(str) +'_' + df['effect_bin'].astype(str)
    x = df.both_bins.value_counts()
    
    #remove bins that contain less simulated :"subjects" than bias groups
    bins_to_remove = x.index[x<=len(bias_dict)].tolist() 
    print('dropping {} bins from bias sampling step'.format(len(bins_to_remove)))
    df_to_split = df[~df['both_bins'].isin(bins_to_remove)]
    return df_to_split


def get_frac_list(scannerDict):
    # get cumulative distirbution values for each scanner's representation of the dataset
    frac_list = []
    for i, (k,v) in enumerate(bias_dict.items()):
        if i==0:
            frac_list.append(v[0])
        else:
            newVal = frac_list[i-1] + v[0]
            frac_list.append(newVal)
    frac_list = frac_list[:-1]
    return frac_list



def get_undersample_lists(scannerDict):
    #calculate % to drop from either class to reach target representation of disease class per scanner
    D_drop_from_class=[]
    ND_drop_from_class=[]
    drop_frac_list=[]
    for i, (k,v) in enumerate(bias_dict.items()):
        D_frac = v[1]
        if D_frac==0.5:
            ND_drop_from_class.append(False)
            D_drop_from_class.append(False)
            drop_frac_list.append(None)
        elif D_frac<0.5:
            ND_drop_from_class.append(False)
            D_drop_from_class.append(True)
            y=D_frac/(1-D_frac)
            drop_frac_list.append(y)
        elif D_frac>0.5:
            ND_drop_from_class.append(True)
            D_drop_from_class.append(False)
            y=(1-D_frac)/D_frac
            drop_frac_list.append(y)
    return D_drop_from_class, ND_drop_from_class, drop_frac_list


def get_bias_stratified_classes(df_to_split, frac_list, drop_from_class, drop_frac_list, disease_class):
    '''
    stratified split of disease and subject effects into bias groups for a given class
    
    df_to_split: df for a specific class containing values for each image's disease and subject effects
    frac_list: list of values corresponding to cumulative distribution for each bias group's representation of the dataset
    drop_from_class: boolean list corresponding to whether a bias subgroup needs to be undersampled for this class
    drop_frac_list: list of correspinding to target representation of this class for each bias group
    disease_class: label for this class (1=disease, 0=no disease)
    
    '''
    
    # 1. split all samples by effect + subject bin value
    
    # here, we create a dictionary for storing separate pieces of df_to_split corresponding to each value of "both_bins"
    bin_names = df_to_split.both_bins.unique()
    binDict = {elem : pd.DataFrame() for elem in bin_names}
    for key in binDict.keys():
        binDict[key] = df_to_split[:][df_to_split.both_bins == key]
    
    
    
    # 2. within each bin, perform stratified split corresponding to bias subgroup representation
    
    splitDict = {} #dictionary for storing dataframes that undergo stratified splitting within each bin
    
    # split the df for each bin into chunks corresponding to the size of each bias group defined in frac_list
    for i, (k,v) in enumerate(binDict.items()):
        lst = np.array_split(v, [int(np.round(m * len(v))) for m in frac_list]) 
        splitDict.update({k:lst})
    
    # undersample within bias groups to get desired representation of bias group within this disease class
    for i, (k,v) in enumerate(splitDict.items()): #i.e. for each bias group within each composite bin...
        lst_keep=[]
        for j in range(len(drop_frac_list)): 
            if drop_from_class[j]==True:
                lst_keep.append(np.array_split(v[j], [int(len(v[j])*drop_frac_list[j])])[0]) #store first element of split df for this scanner (corresponds to proportion of df to keep)
            else:
                lst_keep.append(v[j]) #if not dropping for this scanner, just keep original df 
        splitDict.update({k:lst_keep})
            
    
    splitList = list(splitDict.values()) #list of lists containing split dfs
    
    
    # 3. concatenate and label samples for each bias subgroup
    subgroupList = []
    for i, (k,v) in enumerate(bias_dict.items()):
        df_temp = pd.concat([item[i] for item in splitList], axis=0) #concat each column of list of lists (each col corresponds to a bias subgroup)
        df_temp['bias_label'] = k 
        df_temp['class_label'] = disease_class
        subgroupList.append(df_temp)
        
        
    # 4. combine all scanners into one df for this class  
    df_combined = pd.concat(subgroupList, axis=0)
    sns.jointplot(data=df_combined, x='isv_dst', y='effect_dst', hue='bias_label', palette='Dark2')
    plt.suptitle(f'Disease Class: {disease_class}')
    plt.show()
    
    return df_combined


def train_val_test_split(df, val_frac, test_frac, isv):
    '''
    get train/val/test splits
    '''
    if isv == True:
        df['strat_col'] = df['class_label'].astype(str) +'_' + df['bias_label'].astype(str) + '_' + df['effect_bin'].astype(str) + '_' + df['isv_bin'].astype(str)
    else:
        df['strat_col'] = df['class_label'].astype(str) +'_' + df['bias_label'].astype(str) + '_' + df['effect_bin'].astype(str)
    
    #remove datasets with <=2 from main df
    x = df.strat_col.value_counts()
    bins_to_remove = x.index[x<=2].tolist()
    print('dropping {} bins during training split'.format(len(bins_to_remove)))
    df_to_split = df[~df['strat_col'].isin(bins_to_remove)]
    
    #stratified train/val/test splits
    train, valtest = train_test_split(df_to_split, test_size = test_frac + val_frac, stratify=df_to_split[['strat_col']], random_state=42)
    
    #remove datasets with <=1 from valtest -> to add back to val
    y = valtest.strat_col.value_counts()
    bins_to_remove = y.index[y<=1].tolist()
    print('dropping {} bins during validation split'.format(len(bins_to_remove)))
    valtest_to_split = valtest[~valtest['strat_col'].isin(bins_to_remove)]
    
    val, test = train_test_split(valtest_to_split, test_size = (test_frac/(test_frac+val_frac)), stratify=valtest_to_split[['strat_col']], random_state=42)
    
    print('Total samples after stratification: {}'.format(len(train) + len(val) + len(test)))
    
    return train, val, test



#%%
#bias dict -> first value = proportion of total dataset coming from that group, 
#                second value = overall fraction of that group that belongs to disease class
bias_dict = {
                0: [0.290, 0.845],
                1: [0.210, 0.5],
                2: [0.210, 0.5],
                3: [0.290, 0.155]
                }

    
frac_list = get_frac_list(bias_dict)
drop_from_class_D, drop_from_class_ND, drop_frac_list = get_undersample_lists(bias_dict)

#get subject effect distributions for disease and non-disease classes
isv_nd, isv_d = get_subject_distributions(SEED_S, MU_S, SD_S, TOTAL_N, D_FRAC, LOWER_S, UPPER_S, BINS_S)
NUM_D = len(isv_d)
NUM_ND = len(isv_nd)


df_to_split_D = get_class_distributions(SEED_D, MU_D, SD_D, NUM_D, isv_d['isv_dst'].values, isv_d['isv_bin'].values, BINS_D, BINS_S, 1)
df_to_split_ND = get_class_distributions(SEED_ND, MU_ND, SD_D, NUM_ND, isv_nd['isv_dst'].values, isv_nd['isv_bin'].values, BINS_D, BINS_S, 0)


df_D = get_bias_stratified_classes(df_to_split_D, frac_list, drop_from_class_D, drop_frac_list, 1)
df_ND = get_bias_stratified_classes(df_to_split_ND, frac_list, drop_from_class_ND, drop_frac_list, 0)

df = merge_shuffle_data([df_D, df_ND])


train, val, test = train_val_test_split(df, VAL_FRAC, TEST_FRAC, ISV)
# train.to_csv(save_dir + '/train_dst.csv')
# val.to_csv(save_dir + '/val_dst.csv')
# test.to_csv(save_dir + '/test_dst.csv')


####################
# #figures
# sns.histplot(data=df, x='isv_dst', kde=True, color='darkred')
# plt.xlabel('Subject Effects')
# Path(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/').mkdir(parents=True, exist_ok=True)


# sns.jointplot(data=df, x='isv_dst', y='effect_dst', hue='class_label', palette='cubehelix')
# plt.xlabel('Subject Effects')
# plt.ylabel('Disease Effects')
# plt.suptitle('All Subjects')
# plt.tight_layout()
# plt.subplots_adjust(top=0.95) 
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'classes.png', dpi=300)
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'classes.svg')

# sns.jointplot(data=df.loc[df['class_label']==1], x='isv_dst', y='effect_dst', hue='bias_label', palette='Set1')
# plt.suptitle('Class label = 1')
# plt.xlabel('Subject Effects')
# plt.ylabel('Disease Effects')
# plt.tight_layout()
# plt.subplots_adjust(top=0.95) 
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'class1.png', dpi=300)
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'class1.svg')


# sns.jointplot(data=df.loc[df['class_label']==0], x='isv_dst', y='effect_dst', hue='bias_label', palette='Set1')
# plt.suptitle('Class label = 0')
# plt.xlabel('Subject Effects')
# plt.ylabel('Disease Effects')
# plt.tight_layout()
# plt.subplots_adjust(top=0.95) 
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'class0.png', dpi=300)
# plt.savefig(f'/home/emma/Documents/SBB/figures/stratified distributions figure/{EXP}/' + 'class0.svg')


