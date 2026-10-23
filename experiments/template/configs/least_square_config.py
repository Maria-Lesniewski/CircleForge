# This is a file for storing the default configuration with the trainer. 
## All filepaths are relative to the artifacts directory
import torch.nn as nn
import torch

CONFIG = {
        "data" : {
            "final_annotations_path" : "template_example_experiment/final_data/" + "train_val_test_split_annotations.csv", # adjust to prior split_config file
            "target_columns" : ["ncircles"], # list["str"]
            "feature_transform" : lambda x: x/256, # function obj
            },
        "training" : {
            "logging_path" : "template_example_experiment/", # direct_linear creates folder LinearLeastSquaresResults, we just provide the root
            },
        }

