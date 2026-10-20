# @author Maria Lesniewski
# @file model_config.py
# @desc This is a class for bundling models with their training path/stats

from pathlib import Path
import json
from models.model_options import ModelFactory
import torch

class ModelConfig:
    """
    Models can either be created new from random intial conditions or from checkpoints 
    of other models. This Class loads the required data for loading from checkpoints
    to continue training / logging info that can be logged during training to track various
    iterations

    The Logger class saves model meta data with _save_metadata and formats a JSON file:
    {
    "model_id" : <contains model class name>
    "stats" : <training epoch meta stats>,
    "checkpoint_path" : <abs path to torch.pt file for loading data"
    }
    """
    def __init__(self, model_type : str = None, src : Path =None):
        if ((model_type is None) == (src is None)):
            raise ValueError("ModelConfigurations currently do not support changing model types from a loaded config file. Accordingly pass either a model_type or a config file that contains a model type.")

        # LOAD DATA ABOUT THE MODEL
        self.model_meta = {}
        if src is None:
            self.model_type = model_type
            print(f"New {model_type} model config initialized")
            self.model_meta["src_config_file"] = "New"
            self.model_meta["src_previous_data"] = None
        else:
            self.model_meta["src_config_file"] = str(src)
            with open(src, "r") as f:
                prior_data = json.load(f)
                print(prior_data)
            
            self.model_meta["src_previous_data"] = prior_data["stats"]
            self.model_meta["src_previous_pt"] = prior_data["checkpoint_path"]
            self.model_type = prior_data["model_id"]
            print(f"Continued {model_type} model config intialized from\n {src}")

        # CREATE ACTUAL MODEL 
        self.model = ModelFactory.create(self.model_type, {})
        if (src is not None):
            path = Path(self.model_meta["src_previous_pt"])
            print(f"Loading model parameters from :\n {path}")
            self.model.load_state_dict(torch.load(path))

    def extract_config_step_info(self) -> int:
        """
        Checks the meta data for filepath info indicating batch_id model was obtained from
        """
        if (self.model_meta["src_config_file"] != "New"):
            print("RETURNING SPLIT")
            # checkpoint Filenames are like path/path/path/word_word_word_step#.pt
            print(self.model_meta["src_previous_pt"])
            return int(self.model_meta["src_previous_pt"].split("_")[-1].split(".")[0])
        else:
            return 0

    def is_new(self) -> bool:
        """
        shortcut to find out if model is continuation of previous training or not
        """
        return (self.model_meta["src_config_file"] == "New")


    def to_dict(self) -> dict[str:str]:
        """
        Function to help serialization of object to saved file
        """
        
        return {
                attr_name : attr_value
                for attr_name, attr_value in vars(self).items()
                }


