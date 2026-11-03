# This is a file for storing the default configuration with the trainer. 
## All filepaths are relative to the artifacts directory
import torch.nn as nn
import torch

CONFIG = {
        "data" : {
            "final_annotations_path" : "test_experiment/final_data/" + "train_val_test_split_annotations.csv", # adjust to prior split_config file
            "target_columns" : ["ncircles"], # list["str"]
            "feature_transform" : lambda x: x/256, # function obj
            "batch_size" : 6, # int
            "num_workers" : 1, # int, adjust to num procs available
            },
        "training" : {
            "loss" : nn.MSELoss(), # instance constructor call,
            "optimizer" : torch.optim.Adam, # CLASS (NOT CALL),
            "optimizer_params" : {"lr" : 0.001}, # args outside of model.parameters()
            "logging_path" : "test_experiment/train_stats",
            "epoch_name" : "best_checkpoint_epochs", # Can be Ask or register of epochs,
            "epoch_params" : {"n_epochs" : 10, "validation_interval": 1},
            },
        }

