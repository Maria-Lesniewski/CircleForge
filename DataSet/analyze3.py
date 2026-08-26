import DoodleData as dd 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import pandas as pd 
import analysis_suite as asu
import os

### Data sets
train_data = dd.CircleData(
    csv_file="samples/split_annotations.csv",
    target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units", 
        "fill_fraction"
    ],
    split_tag ="train"
)
val_data = dd.CircleData(
    csv_file="samples/split_annotations.csv",
    target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units",
        "fill_fraction"
    ],
    split_tag ="val"
)

test_data = dd.CircleData(
    csv_file="samples/split_annotations.csv",
    target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units",
        "fill_fraction"
    ],
    split_tag ="test"
)

### Set Analysis 
labels_map = {
        "ncircles" : 0,
        "nbig" : 1,
        "nsmall" : 2 ,
        "area_of_circles" : 3,
        "area_of_negative" : 4,
        "max_circle_units" : 5,
        "fill_fraction" : 6
        }

imgpaths = ["train", "test", "val"]
datmap = { "train" : train_data, "val" : val_data, "test" : test_data}
for imgpath in imgpaths:
    file_exists = os.path.exists(imgpath)
    if not file_exists:
        os.mkdir(imgpath)
    for label in labels_map:
        print(f"Folder {imgpath} : tag {label}")
        asu.print_9_tiles(datmap[imgpath], column=label, lmap=labels_map)
        os.rename(f"{label}.png", f"{imgpath}/{label}.png")
        asu.visualize_histograms(datmap[imgpath], labels_map, f"{imgpath}/histogrammed_data.png")
