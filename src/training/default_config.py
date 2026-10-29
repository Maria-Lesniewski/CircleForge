# This is a file for storing the default configuration with the trainer. 
## All filepaths are relative to the artifacts directory
import torch.nn as nn
import torch

DEFAULT_CONFIG = {
        "data" : {
            "final_annotations_path" : "final_data/train_val_test_split_annotations.csv", # adjust to 
            "target_columns" : ["ncircles"], # list["str"]
            "feature_transform" : None, # function obj
            "batch_size" : 256, # int
            "num_workers" : 4, # int, adjust to num procs available
            },
        "training" : {
            "loss" : nn.MSELoss(), # instance constructor call,
            "optimizer" : torch.optim.Adam, # CLASS (NOT CALL),
            "optimizer_params" : {"lr" : 0.001}, # args outside of model.parameters()
            "logging_path" : "train_stats",
            "epoch_name" : "best_checkpoint_epochs", # Can be Ask or register of epochs,
            "epoch_params" : {"n_epochs" : 100, "validation_interval": 10}, 
            },
        }
