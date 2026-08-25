import pandas as pd 
import matplotlib.pyplot as plt
import DoodleData as dd
import torch

def print_9_tiles(dataset : dd.CircleData, column : str, lmap : dict):
    cols, rows = 3,3
    figure = plt.figure(figsize=(cols*rows-1, cols*rows-1))
    for i in range(1, cols*rows +1):
        samples_idx = torch.randint(len(dataset),size=(1,)).item()
        img, label = dataset[samples_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{column} =\n {label[lmap[column]]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.savefig(f"{column}"+ '.png')
