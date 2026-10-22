# @author Maria Lesniewski
# @file model_config.py
# @desc This is a class for bundling models with their training path/stats

from pathlib import Path
import json
from models.model_options import ModelFactory
import torch
import torch.nn as nn
from typing import Any
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


    Attributes:
        -model_type : str - The kind of model bundled into the config (usually registered with ModelFactory)
        -model : nn.Module - The actual model instance bundled with its info 
        -meta_data : dict -
            - "src_config_file" : path to file with metadata stats about the model
            - "src_previous_data" : dict of model's performance statistics (for extracting during 
                                    epoch routines/ custom workflows)
            - "src_previous_pt" : path to a pt file that can be loaded to create model parameters
    """
    def __init__(self, model_type : str = None, src : Path =None):
        """
        Creates a configuration structure from scratch or from previously trained model
        config stores info about where the model was loaded from and its previous statistics
        (for continuing or starting a training cycle)
        """
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
    
    @classmethod
    def create_from_module(cls, model_type : str, model_params_path : Path = None, model : nn.Module=None, build_model_args : dict[str,Any] = None):
        """
        This is a method for creating a modelconfig module after training has already happened, 
        when no meta data about the model has been previously recorded. 
        (e.g. for taking a model trained outside the CircleForge Trainer class and plugging it 
        into the framework, whether a .pt file for it exists or not )

        Note that if the model is not provided, then the type is assumed to come from one of the
        circleforge registered models. 
        the parameters passed in the model_params_path will be saved into the model 
        the parameter path passed should be absolute filepaths
        """

        config_instance = cls.__new__(cls)
        config_instance.model_type = model_type
        config_instance.model_meta = {}
        config_instance.model_meta["src_previous_data"] = None
        config_instance.model_meta["src_previous_config"] = None

        if (model is None) and (build_model_args is None):
            raise ValueError( "Provide either an existing model (model) or parameters for constructing it via ModelFactory (build_model args) " )

        # Load model 
        if (model is not None):
            config_instance.model = model
        else: 
            config_instance.model = ModelFactory.create(model_type, build_model_args)
        
        # Store params if available

        if (model_params_path is not None):
            path = Path(model_params_path)
            print(f"Loading model parameters from :\n {path}")
            config_instance.model.load_state_dict(torch.load(path))
            config_instance.model_meta["src_previous_pt"] = path
        else:
            config_instance.model_meta["src_prevoious_pt"] = None

        return config_instance
        



