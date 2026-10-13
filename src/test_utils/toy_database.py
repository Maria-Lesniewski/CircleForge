# @author Maria Lesniewski
# @file generate_training_data.py
# @desc This is a class for generating toy databases mid-test (avoids database relative path problem)

import os
from ProceduralGenerator.circle_doodle import CircleDoodle
import numpy as np
from pathlib import Path 
import shutil 

class ToyDatabase:
    """
    This class is for creating / deleting a deterministic 
    toy database employed in unit testing
    """

    def __init__(self, root_path : str):
        # Runtime Path variables
        self.root_path = Path(root_path)
        self.annotations_path = self.root_path / "toy_annotations.csv"
        self.sample_path = self.root_path / "samples"

        # Default Parameters for Generation Conditions
        # Size = len(fill_densities)*len(max_N)*samples_per_setting
        self.seed = 10 
        self.fill_densities = [1]
        self.max_N = list(range(2,10))
        self.samples_per_setting = 30


    def generate(self):
        # Create filepaths to work with
        self.sample_path.mkdir(parents=True, exist_ok=True)

        # Generate Data, set seed for reproducibility
        np.random.seed(self.seed)
        for sample_no in range(0,self.samples_per_setting):
            for max_N in self.max_N:
                for pixel_density in self.fill_densities:
                    writefilename = self.sample_path / f"Nmax_{max_N}_p{pixel_density}_{sample_no}.npy"
                    doodle_obj = CircleDoodle(size=50, 
                                              max_circle = max_N,
                                              fill_percent = pixel_density,
                                              filename=writefilename,
                                              annotations=self.annotations_path
                                              )
                    doodle_obj.fill_space()
                    # print(f"writing {writefilename}")
                    doodle_obj.add_to_data_base(rootdir=None)


    def remove(self):
        # Removes the generated csv file and all data samples
        if (self.sample_path.exists()):
            shutil.rmtree(self.sample_path)
        if (self.annotations_path.exists()):
            self.annotations_path.unlink()