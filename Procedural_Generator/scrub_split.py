# @author Maria Lesniewski
# @file scrub_split.py
# @desc This is a driver for quality controlling the generated data set / splitting it
# @desc Generally, we don't want duplicates in our data base and we want to split the data according to some train, val, test split
# @desc The generation doesnt do that. This script scrubs the master list, scrambles it, then adds split labels. 

import generatorlib as gen
import numpy as np
import os
from pathlib import Path
from mpi4py import MPI
import pandas as pd


### Skeleton 
### Search Data Base for matches on all columns. If all columns match, load .npy file and check if identical. If Identical, delete line of one of duplicates from CSV, remove from directory 
### Then randomly shuffle and add labels for train, val, test etc. 


# Part 1: Search 
## load .csv 
df = pd.read_csv("samples/annotations.csv")

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

                        """ DEBUGGING
                        image1 = gen.CircleDoodle()
                        image2 = gen.CircleDoodle()
                        image1.load_circle_obj_from_data("samples/annotations.csv", image_path)
                        image2.load_circle_obj_from_data("samples/annotations.csv", partner_path)
                        image1.show_obj()
                        image2.show_obj()
                        input("Quit")
                        exit(1)
                        """

                        break

files_to_delete = []
print(drop_idxs)
for idx in drop_idxs:
    files_to_delete.append(df.iloc[idx]["access_filename"])

for idx, file in zip(drop_idxs, files_to_delete):
    Path(file).unlink()
    df = df.drop(idx)

df.reset_index(drop=True)

### Part 2 : Split

