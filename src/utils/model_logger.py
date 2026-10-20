# @author Maria Lesniewski
# @file model_logger.py
# @desc This is a class for bundling typical logging options
# @desc It exists so that relative training paths can be handled in scripts directly
# @desc and also different file formats for humans/machines can be processed fast

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Any
import json
import os 
class Logger:
    """
    This is a class for recording experimental details 
    using text files and tensorboard, alongside checkpointing

    note that because this class uses torch's summarywriter
    loggerobject.close() must be called
    """
    def __init__(self, root_path : Path):
        self.root_path = root_path
        self.torchwriter = SummaryWriter(root_path / "tensorboard")
        self.grace_path = root_path / "grace"
        self.grace_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = root_path / "checkpoints"
        self.meta_path = root_path / "cpt_meta_data"

    def add_scalars(self,
            tag : str, # label the scalar sets we're adding (by process or by qty)
            scalars : dict[str, float], # custom filenames / what we're evaluating on
            step): # The epoch or the batch we're saving for
        """
        For appending x y points to files for later analysis
        """
        for name, scalar in scalars.items():
            grace_file_path = self.grace_path / f"{tag}_{name}.xvg"
            grace_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(grace_file_path, "a") as f:
                np.savetxt(f, [[step, scalar]])

        self.torchwriter.add_scalars(tag, scalars, step)

    def close(self):
        self.torchwriter.flush()
        self.torchwriter.close()

        # remove empty directories we did not write to 
        # (tensorboard always writes an event log, grace doesn't)
        if os.path.isdir(self.grace_path) and not os.listdir(self.grace_path):
            os.rmdir(self.grace_path)

    def save_model(
            self, 
            model_id : str, 
            step : int, # batch step or epoch saved
            model : torch.nn.Module, # Actual model to save
            stats : dict[str, Any], # meta statistics
            metadata : bool # whether to save a json with metastatistics
            ):
        path = self.checkpoint_path / f"{model_id}_step_{step}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        if metadata:
            self._write_model_metadata(model_id, step, stats)

    def _write_model_metadata(
            self,
            model_id : str, # name of the model
            step : int, # iteration stats are saved for
            stats : dict[str, Any]):
        path = self.meta_path / f"{model_id}_step_{step}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model_id" : model_id, 
                "stats" : stats, 
                "checkpoint_path" : str(self.checkpoint_path / f"{model_id}_step_{step}.pt")}
        with open(path, 'w') as f:
            json.dump(data, f, default=str, indent=4)

    def write_json(self, rel_filepath : str, dictionary : dict[Any, Any]):
        """
        Saves any python dictionary as a .json file in the rootfile path
        """
        path = self.root_path/ rel_filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        if (path.exists()):
            print(f"Warning: {path} exists")
            copy_path = path.with_name(path.stem +  ".backup" + path.suffix)
            print(f"Creating backup file {copy_path}")
            self.copy_json(path)
        with open(path, 'w') as f:
            json.dump(dictionary, f, default=str, indent=4)

    def copy_json(self, abs_filepath):
        """
        Creates a copy of the file passed in the abs_filepath
        Checks to make sure thats not overwriting another file
        """
        copy_path = abs_filepath.with_name(abs_filepath.stem +  ".backup" + abs_filepath.suffix)
        if copy_path.exists():
            print(f"{copy_path} exists")
            copy_copy_path = copy_path.with_name(copy_path.stem +  ".backup" + copy_path.suffix)
            print(f"creating backup {copy_copy_path}")
            self.copy_json(copy_path)

        with open(abs_filepath, 'r') as f:
            data_to_copy = json.load(f)

        with open(copy_path, 'w') as f:
            json.dump(data_to_copy, f, default=str, indent=4)




