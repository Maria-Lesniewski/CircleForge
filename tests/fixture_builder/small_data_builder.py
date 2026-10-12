# @author Maria Lesniewski
# @file generate_training_data.py
# @desc This is the driver for creating procedurally generated circle drawings in a data base.
import os
import utils.io as io
from ProceduralGenerator.circle_doodle import CircleDoodle
import numpy as np
import random
import pandas as pd
from pathlib import Path 

### Data Generation Parameters 
negative_space_fill_densities = [1]
max_N_to_try = list(range(2,10))
#np.array([1]) # list(range(0, 200))
samples_per_setting = 30

fix_path = io.find_here() / "fixture"
annotations_path = fix_path / "toy_annotations.csv"
fix_path.mkdir(parents=True, exist_ok=True)
sample_path = fix_path / "samples"
sample_path.mkdir(parents=True, exist_ok=True)
# Generate Data, set seed for reproducibility
np.random.seed(10)
random.seed(10)

for sample_no in range(0,samples_per_setting):
    for max_N in max_N_to_try:
        for pixel_density in negative_space_fill_densities:
            writefilename = sample_path / f"Nmax_{max_N}_p{pixel_density}_{sample_no}.npy"
            doodle_obj = CircleDoodle(size=50, max_circle = max_N, fill_percent = pixel_density, filename=writefilename, annotations=annotations_path)
            doodle_obj.fill_space()
            print(f"writing {writefilename}")
            doodle_obj.add_to_data_base(rootdir=None)
