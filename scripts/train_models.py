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

# Load Data Into Data Set
print("Loading training/validation data samples")
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = CircleData(
        csv_file=data_fp, 
        target_columns=["ncircles"],
        split_tag="train"
        )
val_data = CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="val"
        )

# Place in Data Loader
## Loaders
trloader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=4
        )
val_loader = DataLoader(
        val_data,
        batch_size=256,
        shuffle=True,
        num_workers=4
        )
print("Created data loaders.")

model_config = None
step = None
if len(sys.argv) > 1:
    print("DETECTED ARGS AFTER python train_model.py")
    print("ASSUMING IT CORRESPONDS TO A METADATA PATH")
    model_config = ModelConfig(src=Path(sys.argv[1]))
    print("Model configuration loaded successfully")
    step = model_config.extract_config_step_info()
    print(f"obtained step {step}")
else:
    ## Model Selection for first round training
    model_key = io.multiple_choice_q("The following model architectures are implemented", list(mo.ModelFactory._registry.keys()))
    model_config = ModelConfig(model_type=model_key)

## Loss Function 
loss_crit = nn.MSELoss()
print(f"Using losscrit: {type(loss_crit).__name__}")
learning_rate = {"learning_rate" : 1e-3}
if len(sys.argv) > 1: 
    while True:
        learning_rate = io.dictionary_filling_question("Detected model path, meaning this may be round two+ of training. Please enter a learning rate. ", ["learning_rate"])
        if is_float(learning_rate["learning_rate"]):
            break
        else: 
            print("Learning rate must be float. Try again.")

optimizer = torch.optim.Adam(list(model_config.model.parameters()), lr=float(learning_rate["learning_rate"]))

print(f"Using optimizer: {type(optimizer).__name__}")

## Training Algorithm Selection (Currently sort of a toy Sanity Debug) 
epoch_key = io.multiple_choice_q("The following epoch algs are active", list(TL.Trainer._epoch_registry.keys()))

# Create a directory to save in based on choices, and create a writer to use while training
root_path = io.artifact_path(f"{model_config.model_type}_{epoch_key}_training")
io.make_artifact_dir(root_path)
print(f"Training logs will be stored in directory {root_path}")
scribe = Logger(root_path)

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
config = {"n_epochs" : 100, "validation_interval" : int(len(trloader)/10)}
for _ in range(0,3):
    print("####TRAINING#######")

model_trainer.train(config)

print("Training complete. Closing logger")
scribe.close()
