# This file contains default settings for creating a dataset 
# using iterative calls to the generator CircleDoodle class
# Its purpose is to function as a namespace seperation for when the 
# mainline generate_data.py script has parameters or not. 
# All filepaths listed are relative to the root project directory's artifacts folder. 

DEFAULT_CONFIG = {
        "fill_frac" : [0.1, 0.01, 0.3, 0.5, 1],
        "max_N_to_try" : list(range(0,200)),
        "samples_per_setting" : 50,
        "data_root_filepath" : "raw_data"
        } # annotations files are saved under artifact / <data_root_filepath> / as combined_raw_annotations.csv. (per processor annotations stored as processor_n_annotations.csv
# Actual images stored as .npy files in artifacts/ <data_root_filepath> / samples
