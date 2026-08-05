# @author Maria Lesniewski
# @file main.py
# @desc This is the driver for my silly physics based diffusion experiment

import generatorlib as gen
import numpy as np
import os


# Generate Data, set seed for reproducibility
np.random.seed(7)

file_exists = os.path.exists("data_folder")
if not file_exists:
    os.mkdir("data_folder")

for i in range(0,1000):
    for max in range(0,200, 10):
        writefilename = f"Nmax_{max}_{i}.npy"
        doodle_obj = gen.CircleDoodle(size=600, max_circle = max, fill_percent = 0.01, filename=writefilename)
        doodle_obj.fill_space()
        print(f"writing {writefilename}")
        doodle_obj.add_to_data_base("data_folder")