# @author Maria Lesniewski
# @file train_models.py
# @desc This is a driver for training models that predict N circles using PyTorch
# @desc For now, it will accept a model option and train it accordingly 

import torch # Pytorch
import torch.nn as nn
import models.model_options as mo # Custom Models
import training.trainer as TL # Pre-stored Training epochs 
from utils.model_logger import Logger
from dataset.circle_data import CircleData # Data class for loading into torch
from torch.utils.data import DataLoader # Actual loaders
import utils.io as io # For anchoring relative io paths across modules
from utils.type_checker import is_float
import json
import sys 
from pathlib import Path
from models.model_config import ModelConfig
from training.default_config import DEFAULT_CONFIG

################################
#####       SETUP          #####
################################

config = None
# OBTAIN CONFIG FOR TRAINING
args = io.parse_train_args()
if (args.config is not None):
    config = io.load_config_from_CL(args.config)
else:
    config = DEFAULT_CONFIG


## OBTAIN CONFIG FOR MODEL TO BE TRAINED 
if (args.model is not None):
    # PREVIOUSLY TRAINED MODELS
    model_config = ModelConfig(src=Path(args.model))
else:
    ## NEW MODEL SELECTED FROM USER
    model_key = io.multiple_choice_q(
            "The following model architectures are implemented",
            list(mo.ModelFactory._registry.keys())
            )
    model_config = ModelConfig(model_type=model_key)

# UNPACK CONFIG TRAINING INFO FOR LOADING DATA AND TRAINING
annotations_fp = io.artifact_path(config["data"]["final_annotations_path"])
cols=config["data"]["target_columns"]
tfunc=config["data"]["feature_transform"]
batch=config["data"]["batch_size"]
nworkers=config["data"]["num_workers"]

loss_crit = config["training"]["loss"]
optimizer_args = {"params" : list(model_config.model.parameters()) } | config["training"]["optimizer_params"]
optimizer = config["training"]["optimizer"](**optimizer_args)
epoch_key = config["training"]["epoch_name"]
epoch_args = config["training"]["epoch_params"]
root_save_path = io.artifact_path(config["training"]["logging_path"] ) / f"{model_config.model_type}_{epoch_key}_training"

# CREATE LOG AND DIRECTORY
io.make_artifact_dir(root_save_path)
print(f"Training logs will be stored in directory {root_save_path}")
scribe = Logger(root_save_path)

##############################################
### FINISH SETUP NOW USE INFORMATION AND TRAIN
##############################################
# Load Data Into Data Set
print("Loading training/validation data samples")
train_data = CircleData(
        csv_file=annotations_fp, 
        target_columns=cols,
        split_tag="train",
        transform=tfunc,
        )
val_data = CircleData(
        csv_file=annotations_fp,
        target_columns=cols,
        split_tag="val",
        transform=tfunc
        )

# Place in Data Loader
## Loaders
trloader = DataLoader(
        train_data,
        batch_size=batch,
        shuffle=True,
        num_workers=nworkers,
        )
val_loader = DataLoader(
        val_data,
        batch_size=batch,
        shuffle=True,
        num_workers=nworkers,
        )
print("Created data loaders.")


print("Train Params")
print(f"Using losscrit: {type(loss_crit).__name__}")
print(f"Using optimizer: {type(optimizer).__name__}")
print(f"Using epoch routine : {epoch_key} with args: {epoch_args}")

model_trainer = TL.Trainer(
        model_config=model_config,
        optimizer=optimizer,
        loss_fn = loss_crit,
        train_loader=trloader,
        val_loader=val_loader,
        logger=scribe,
        epoch_style = epoch_key,
        )

print("Created training object")
for _ in range(0,3):
    print("####TRAINING#######")

model_trainer.train(epoch_args)

print("Training complete. Closing logger")
scribe.close()
