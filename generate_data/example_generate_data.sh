#!/bin/bash

# get sampling distributions for disease and subject effects for each split
#   user-defined parameters for defining distributions are defined within this python code.
#   this script drops "bins" containing too few items to be stratified. may need to
#   guess + check on original number of samples (TOTAL_N) to end up with your target dataset size.
#   would recommend running this code separately to make sure you're happy with the final dataset size before actually generating data

python define_stratified_distributions.py

# ----------------

#generate data with previously defined effect distributions for each train/val/test split
exp=experiment_name #experiment name you used in define_stratified_distributions.py

echo generating data for $exp
python generate_data.py --d_roi 102 --b_roi 164 --isv 1 --split train --expname $exp
sleep 60
python generate_data.py --d_roi 102 --b_roi 164 --isv 1 --split val --expname $exp
sleep 60
python generate_data.py --d_roi 102 --b_roi 164 --isv 1 --split test --expname $exp
sleep 60
