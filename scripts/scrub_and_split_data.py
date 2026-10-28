# @author Maria Lesniewski
# @file scrub_and_split_data.py
# @desc This is a driver for quality controlling the generated data set / splitting it
# @desc It removes duplicates and scrambles the samples 
# @desc then we split the data according to some train, val, test split fraction

import sys
import importlib.util
import utils.io as io
import generator.circle_doodle as gen
from dataset.scrub_conditions import DEFAULT_CONFIG
import numpy as np
from pathlib import Path
import pandas as pd

np.random.seed(0)


config = None
args = io.parse_config_args()

if (args.config is not None):
    config = io.load_config_from_CL(args.config)
else:
    config = DEFAULT_CONFIG

### SPLIT PROPORTIONS ###
f_train = config["train_frac"]
f_val = config["val_frac"]
f_test = config["test_frac"]

# FILE PATHS 
raw_path = io.artifact_path(config["raw_rootpath"])
raw_annotations = config["raw_annotations"]
final_data = config["scrubbed_rootpath"]
#### BEGIN SCAN FOR DUPLICATES ###
raw_file_annotations = raw_path / raw_annotations
print(f"Checking file {raw_file_annotations} for duplicate entries")
## load .csv 
df = pd.read_csv(raw_file_annotations)

## Look at all data except access_filename/image_filename columns 
label_columns = [
        col for col in df.columns
        if (col != "access_filename" and col != "image_filename")
        ]

entries_to_delete = []

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

print(f"Found {len(drop_idxs)} sample entries to drop")
for idx in drop_idxs:
    entries_to_delete.append(df.iloc[idx])
### END SCAN FOR DUPLICATES ### 

### DELETE DUPLICATES FROM MANIFEST, Create a record of the removal
deleted_manifest = pd.DataFrame(entries_to_delete).reset_index(drop=True)
for idx in drop_idxs:
    df = df.drop(idx)

df.reset_index(drop=True)
### SHUFFLE REMAINING ###
shuff = np.random.choice(len(df), size=len(df), replace=False)
df = df.iloc[shuff].reset_index(drop=True)

### SPLIT REMAINING DATA INTO TRAIN/VAL/TEST ###
n_samples = len(df)
n_train = int(n_samples*f_train)
n_val = int(n_samples*f_val)
n_test = int(n_samples*f_test)


if (n_train + n_val + n_test != n_samples):
    dif = n_samples - (n_train + n_val + n_test)
    n_val = n_val + dif

split_labels = ["train"]*n_train + ["val"]*n_val + ["test"]*n_test
df["data_split"] = split_labels

### Save Deleted Manifest and Final Data Manifest
fin_art_data_path = io.make_artifact_dir(final_data)
final_split_annotation = fin_art_data_path / "train_val_test_split_annotations.csv"
record_of_deletes = fin_art_data_path / "samples_removed_from_raw.csv"
df.to_csv(final_split_annotation, index=False)
deleted_manifest.to_csv(record_of_deletes, index=False)
print(f"Stored final data manifest in:\n {final_split_annotation}")
print(f"Stored deleted points (duplicates/non-existant files) in:\n {record_of_deletes}")

