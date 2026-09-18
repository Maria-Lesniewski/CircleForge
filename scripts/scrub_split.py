# @author Maria Lesniewski
# @file scrub_split.py
# @desc This is a driver for quality controlling the generated data set / splitting it
# @desc It removes duplicates and scrambles the samples 
# @desc then we split the data according to some train, val, test split fraction

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utils.io as io
import ProceduralGenerator.generatorlib as gen
import numpy as np
from pathlib import Path
import pandas as pd

### SPLIT PROPORTIONS ###
f_train = 0.6
f_val = 0.2
f_test = 0.2

#### BEGIN SCAN FOR DUPLICATES ###
## load .csv 
df = pd.read_csv("raw_data/annotations.csv")

## ignore access_filename/image_filename columns 
label_columns = [
        col for col in df.columns
        if (col != "access_filename" and col != "image_filename")
        ]

rows_to_delete = []
files_to_delete = []

group = df.groupby(label_columns, dropna=False)

drop_idxs = [] # We dont expect too many of these 
for group_key, group_df in group:
    if len(group_df) < 2:
        continue
    else:
        check_idx = 0
        # Check for duplicates in the block 
        for idx, row in group_df.iterrows():
            check_idx= check_idx + 1
            image_path = df.loc[idx]["access_filename"]

            if not Path(image_path).exists():
                print(f"WARNING: missing file: {image_path}")
                drop_idxs.append(idx)

            else:
                image = np.load(image_path)
                # Check i in block greater than we've checked
                for i in range(check_idx, len(group_df)):
                    partner_path = group_df.iloc[i]["access_filename"]
                    partner_image = np.load(partner_path)
                    if (np.array_equal(image, partner_image)):
                        drop_idxs.append(idx)
                        
                        break

files_to_delete = []

for idx in drop_idxs:
    files_to_delete.append(df.iloc[idx]["access_filename"])
### END SCAN FOR DUPLICATES ### 

### DELETE DUPLICATES FROM MANIFEST
for idx, file in zip(drop_idxs, files_to_delete):
    io.delete_file_abs_path(file)
    df = df.drop(idx)

df.reset_index(drop=True)
### SHUFFLE REMAINING ###
shuff = np.random.choice(len(df), size=len(df), replace=False)
df = df.iloc[shuff].reset_index(drop=True)

### SPLIT DATA ###
n_samples = len(df)
n_train = int(n_samples*f_train)
n_val = int(n_samples*f_val)
n_test = int(n_samples*f_test)


if (n_train + n_val + n_test != n_samples):
    dif = n_samples - (n_train + n_val + n_test)
    n_val = n_val + dif

split_labels = ["train"]*n_train + ["val"]*n_val + ["test"]*n_test
df["data_split"] = split_labels

### Save 
io.
final_split_annotation = io.artifact_path("")
df.to_csv('split_annotations.csv', index=False)

