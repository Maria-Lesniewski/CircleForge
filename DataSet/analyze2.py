import DoodleData as dd 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import pandas as pd 
from analysis_suite import print_9_tiles 
dataset = dd.CircleData(
    csv_file="samples/annotations.csv",
    target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units", 
        "fill_fraction"
    ]
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for x, y in loader:
    print(x.shape, y.shape)
    break
labels_map = {
        "ncircles" : 0,
        "nbig" : 1,
        "nsmall" : 2 ,
        "area_of_circles" : 3,
        "area_of_negative" : 4,
        "max_circle_units" : 5,
        "fill_fraction" : 6
        }

for label in labels_map:
    print(label)
    print_9_tiles(dataset, column=label, lmap=labels_map)
