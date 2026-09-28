# @file ModelAnalyzer.py
# @author Maria Lesniewski
# @desc This is a class to abstract model analysis into one callable interface and modular functions to test. 
# @desc Generally, all metrics will require model to evaluate data. Rather than doing that
#       repeatedly, this class registers each modular analysis type and reuses the same predictions
#       as requested by a user to log data / performance
from typing import Dict, Any
import torch
import torch.nn as nn
import utils.model_logger as ml
import utils.io as io
from torch.utils.data import DataLoader
## Layout Plan 

class ModelAnalyzer:
    """
    Coordinates modular analysis routines for a trained model.

    ModelAnalyzer provides a common interface for requesting model
    analysies and dispatches requests to registered analysis routines.

    Class Attributes:
        _function_registry : Maps analysis strings to appropriate function implemented 
    """
    _function_registry : Dict[str,Any] = {
            # Analysis registry to be filled via implementation below
            }
    def register_analysis(analysis_name: str, registry=_function_registry):
        def decorator(analysis_function):
            """
            decorator factory for adding to dictionary of implemented analysis routines
            """
            registry[analysis_name] = analysis_function
            return analysis_function
        return decorator


    def __init__(self, model : nn.Module , data_set_loader : DataLoader, filewriter_root_path : str):
        """
        Obj Attributes:

            model : nn.Module - Usually loaded from a .cpt then passed to this init
            data_set_loader : DataLoader - Set for which evaluation should happen
            filewriter_root_path : str - root path that all analysis will be saved under in sub folders

        """
        self.model : nn.Module = model
        self.dataloader : DataLoader = data_set_loader
        self.filewriter_root_path : str = filewriter_root_path
        self.calculation_schedule : Dict[str,Any] = {}
        self.calculation_config : Dict[str, Any] = {}


    def obtain_schedule_and_configs(self):
        """
        Prints list of all possible implemented analysis, 
        if analysis has options, asks for relevant options 
        """
        intro_context = ("The Model Analyzer is used to evaluate pre-trained models\n" +
                "The following are currently implemented analysis routines\n")
        io.compounding_multiple_choice_q(intro_context, self._function_registry)


    @register_analysis("RMSD")
    def RMSD(self):
        print("Hello")

## Register of Functions
