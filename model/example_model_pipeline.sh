#!/bin/bash

#run full model pipeline
exps=(experiment_name1 experiment_name2 experiment_name3)
arrs=(TP TN)

for exp in "${exps[@]}"
do

      echo training model for $exp
      python model_3d.py --expname $exp #train model

      echo evaluating $exp
      python evaluate_3d.py --exp_name $exp #evaluate model

      for arr in "${arrs[@]}"
      do

              echo generating saliency for $exp $arr
              python xai_3d.py --exp_name $exp --array $arr --bias_label 0 #saliency maps for non-bias group
              python xai_3d.py --exp_name $exp --array $arr --bias_label 1 #saliency maps for bias group
              python avg_saliency_maps.py --exp_name $exp --array $arr --bias 1 #average saliency maps
              python saliency_ROIs.py --smap <path_to_avg_saliency_map> --expname $exp --group $arr --bias_label 0 #quantitative saliency scores for non-bias group
              python saliency_ROIs.py --smap <path_to_avg_saliency_map> --expname $exp --group $arr --bias_label 1 #quantitative saliency scores for bias group
      done

done
