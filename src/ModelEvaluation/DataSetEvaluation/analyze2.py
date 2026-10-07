import DoodleData as dd 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import pandas as pd 
import analysis_suite as asu
import os
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

image_path = "Image_Examples_by_datalabel"
file_exists = os.path.exists(image_path)
if not file_exists:
    os.mkdir(image_path)


for label in labels_map:
    print(label)
    asu.print_9_tiles(dataset, column=label, lmap=labels_map)
    os.rename(f"{label}.png", f"{image_path}/{label}.png")

# asu.visualize_histograms(dataset, labels_map, f"{image_path}/histogrammed_data.png")

train, val, test = asu.split_database(dataset, 60, 20, 20)

asu.visualize_histograms(train, labels_map, f"{image_path}/training_hist.png")




