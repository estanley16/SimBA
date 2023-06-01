#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:46:28 2022

@author: emma
"""

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

def get_dataset_info(directory, class_label):
    '''
    get dataframe of dataset information
    input -> path to folder with images, desired class label
    output -> df with filename, path, class label, and bias label
    '''
    #get filenames
    file_list = os.listdir(directory)

    #get full filepaths
    file_paths=[]
    bias_label_list = [0]*len(file_list)
    for i,f in enumerate(file_list):
        # print(f)
        g = directory + '/' + f
        file_paths.append(g)

        #get list of bias labels for this directory
        if f[-8]=='B': #if files is appended with "B"
            bias_label_list[i]=1

    #get list of class labels for this directory
    class_label_list = [class_label]*len(file_list)

    #make a dataframe with all the information
    df = pd.DataFrame(list(zip(file_paths, file_list, class_label_list, bias_label_list)), columns=['filepath', 'filename', 'class_label', 'bias_label'])
    df['class_label'] = pd.to_numeric(df['class_label'])
    df['bias_label'] = pd.to_numeric(df['bias_label'])

    return df

def merge_shuffle_data(df_list):
    '''
    input -> list of dataframes to combine
    output -> dataframe with all data combined and randomly shuffled
    '''
    full_df = pd.concat(df_list)
    full_df = full_df.sample(frac=1, random_state=6)

    return full_df

def train_test_split_save(full_df, save_dir, test_frac, val_frac):
    '''
    input -> df with all data, directory to save splits to, test and validation fractions of full dataset
    output -> train, val, test splits
    '''
    train, valtest = train_test_split(full_df, test_size = test_frac + val_frac, stratify=full_df[['class_label', 'bias_label']])
    val, test = train_test_split(valtest, test_size = (test_frac/(test_frac+val_frac)), stratify=valtest[['class_label', 'bias_label']])

    train.to_csv(save_dir + '/train.csv')
    val.to_csv(save_dir + '/val.csv')
    test.to_csv(save_dir + '/test.csv')

    return train, val, test

def main():

    # #use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, help='experiment name', required=True)
    args = parser.parse_args()
    exp_name = args.expname

    # exp_name = 'exp25'

    D_dir = '/home/emma/Documents/SBB/datasets/' + exp_name + '_D'
    ND_dir = '/home/emma/Documents/SBB/datasets/' + exp_name + '_ND'

    save_dir = '/home/emma/Documents/SBB/' + exp_name
    #create directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)


    df_D = get_dataset_info(D_dir, class_label=1)
    df_ND = get_dataset_info(ND_dir, class_label=0)


    df = merge_shuffle_data([df_D, df_ND])

    train, val, test = train_test_split_save(df, save_dir, test_frac=0.3, val_frac=0.15)

if __name__ == '__main__':
    main()







