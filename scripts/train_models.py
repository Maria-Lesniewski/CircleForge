# @author Maria Lesniewski
# @file train_models.py
# @desc This is a driver for training models that predict N circles using PyTorch
# @desc For now, it will accept a model option and train it accordingly 

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import Models.model_options as mo # Models
import Training.trainer as TL # Pre-stored Training epochs 
import DataSet.DoodleData as dd # Data class for loading into torch
from torch.utils.data import DataLoader # Actual loaders
import utils.io as io # For anchoring relative io paths across modules

from torch.utils.tensorboard import SummaryWriter # For logging 

# Load Data Into Data Set
print("Loading training/validation data samples")
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = dd.CircleData(
        csv_file=data_fp, 
        target_columns=["ncircles"],
        split_tag="train"
        )
val_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="val"
        )

# Place in Data Loader
## Loaders
trloader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
        )
val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
        )
print("Created data loaders.")
## Loggers and log directory
log_path = io.make_artifact_dir("training_log")
writer = SummaryWriter(log_path)

## Model Selection
model_key = io.multiple_choice_q("The following model architectures are implemented", list(mo.ModelFactory._registry.keys()))

model = mo.ModelFactory.create(model_key, config={})
## Training Algorithm Selection (Toy for Sanity Debug) 
epoch_key = io.multiple_choice_q("The following epoch algs are active", list(TL.Trainer._epoch_registry.keys()))

