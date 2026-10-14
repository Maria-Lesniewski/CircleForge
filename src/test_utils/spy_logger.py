# @author Maria Lesniewski
# @file spy_logger.py
# @desc This is a class for dummying a logger without calling actual logging functions
# @desc It exists to make unit testing easier for statistical analysis that also expects runners

import torch
import torchvision # This is here to avoid a segfault from tensorboard
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Any
import json
import os
from utils.model_logger import Logger 

class SpyLogger(Logger):
    """
    This is a Test class (test double?/ spy?) 
    for passing to functions that expect to have logging behavior
    when there is no logging behavior required. The idea is to make whatever is logged directly 
    inspectable via the logger

    Its usually used for unit testing where we inspect data quantities directly instead of in
    written files 
    """
    def __init__(self, root_path : Path = None):
        self.root_path = None # root_path
        self.torchwriter = None #SummaryWriter(root_path / "tensorboard")
        self.grace_path = None # root_path / "grace"
        self.checkpoint_path = None # root_path / "checkpoints"
        self.meta_path = None # root_path / "cpt_meta_data"


        ## Funcs to store as attributes of call dicts
        self.add_scalars_calls = {} # We'll index them by the tag or id used in each. 
        self.write_json_calls = {}
        self.save_model_calls = {}


    def add_scalars(self,
            tag : str = None, # label the scalar sets we're adding (by process or by qty)
            scalars : dict[str, float] = None, # custom filenames / what we're evaluating on
            step = None): # The epoch or the batch we're saving for
        self.add_scalars_calls[tag] = {
                "scalars" : scalars,
                "step" : step
                }

    def close(self):
        pass

    def save_model(
            self, 
            model_id : str, 
            step : int, # batch step or epoch saved
            model : torch.nn.Module, # Actual model to save
            stats : dict[str, Any], # meta statistics
            metadata : bool # whether to save a json with metastatistics
            ):
        self.save_model_calls[model_id] = { 
                "step" : step, 
                "model" : model,
                "stats" : stats,
                "metadata" : meta_data
                }

    def _write_model_metadata(
            self,
            model_id : str, # name of the model
            step : int, # iteration stats are saved for
            stats : dict[str, Any]):
        pass

    def write_json(self, rel_filepath : str, dictionary : dict[Any, Any]):
        self.write_json_calls[rel_filepath] = {"dictionary" : dictionary}

