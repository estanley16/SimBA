#!/bin/bash

labels_path=/home/emma/Documents/SBB/atlas/LPBA40-labels.csv #path to csv file with atlas labels
velo_path=/home/emma/Documents/SBB/IXI_nonlin_velo_fields_reduced #path to folder with velocity fields

#generate masks for each region in label atlas
python save_roi_masks.py

#generate pca model for each region in the label atlas (where region values contained in col1)
while IFS="," read -r col1 col2 col3 col4 col5
do
        echo "generating PCA model for region: $col1"
        python pca_velo_ixi.py --velo_path velo_path --region $col1

done < <(tail -n +2 labels_path)


#generate pca model for global subject morphology / inter-subject variability
python pca_isv_velo_ixi.py --velo_path velo_path --savename isv
