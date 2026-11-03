# This file contains custom settings for creating a dataset 
# using iterative calls to the generator CircleDoodle class
# All filepaths listed are relative to the root project directory's artifacts folder. 

CONFIG = {
        "fill_frac" : [0.1],
        "max_N_to_try" : list(range(0,10)),
        "samples_per_setting" : 1,
        "data_root_filepath" : "template_example_experiment/raw_samples"
        } # annotations files are saved under artifact / <data_root_filepath> / as combined_raw_annotations.csv. (per processor annotations stored as processor_n_annotations.csv
# Actual images stored as .npy files in artifacts/ <data_root_filepath> / samples

