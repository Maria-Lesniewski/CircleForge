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

    Seperates analysis that require total knowledge of the set from running knowledge
    of the set to save on memory / to help generalize to larger prediction tasks

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

    def __init__(self, model : nn.Module , data_set_loader : DataLoader, filewriter_root_path : str, evaluation_objective : str):
        """
        Obj Attributes:

            model : nn.Module - Usually loaded from a .cpt then passed to this init
            data_set_loader : DataLoader - Set for which evaluation should happen
            evaluation_objective : str - The name of the target for which the model should be evaluated against. (i.e. what does the model predict) 
            filewriter_root_path : str - root path that all analysis will be saved under in sub folders
            evluation_objective : str - The ground truth label for evaluating the model against
        """
        self.model : nn.Module = model
        self.evaluate_against = evaluation_objective
        self.dataloader : DataLoader = data_set_loader
        self.filewriter_root_path : str = filewriter_root_path
        self.calculation_schedule : Dict[str,Any] = {}
        self.calculation_config : Dict[str, Any] = {}
        self.calculation_config_options : Dict[str:str] = {}
        self.determine_calc_config_options()

    def determine_calc_config_options(self):
        """
        Inspects data set passed for multiple / correct labels. 
        If multiple are available generates user options for binning analysis by label
        or doing analysis on global set
        or both 
        Also functions as failsafe for if dataloader passed doesnt have analysis objective in it
        """
        if len(self.dataloader.target_columns) != 1:
            self.calculation_config_options["global"] = "Perform calculation wrt. total dataset"
            self.calculation_config_options["local"] = "Perform calculations as a function of each other label in the set" 
            self.calculation_config_options["both"] = "Calculate global performance and local performance"
        if (self.evaluate_against not in self.dataloader.target_columns):
            error_str = (f"target_objective '{self.evaluate_against}' not in data loader\n" +
                    f"Received dataset with columns {self.dataloader.target_columns}")
            raise ValueError(error_str)

    def obtain_schedule_and_configs(self):
        """
        Prints list of all possible implemented analysis, 
        if analysis has options, asks for relevant options 
        """
        intro_context = ("The Model Analyzer is used to evaluate pre-trained models\n" +
                "The following are currently implemented analysis routines\n")
        choices = io.compounding_multiple_choice_q(intro_context, list(self._function_registry.keys()))
        for ele in choices:
            self.calculation_schedule[ele] = self._function_registry[ele]

        if (self.calculation_config_options):
            print("Found multiple labels in data set")
            print("Please specify the data skew for the routines selected")
            context_str = "Options are\n"
            option_str = "\n".join(
                    f"{key}: {description}"
                    for key, description in self.calculation_config_options.items()
                    )
            context_str += option_str
            self.calculation_config = io.multiple_choice_q(context_str, list(self.calculation_config_options.keys()))

    def run_schedule(self):
        """
        Perform and log each calculation listed
        according to the config mode user passed
        """
        for func in self.calculation_schedule.keys():
            if self.calculation_schedule[func].execution == "running":
               self.calculation_schedule[func](self)



#### Begin Grabhat of Function Implementations 
    @register_analysis("RMSE")
    def RMSE(self):
        print("Hello from RMSE")

    RMSE.execution = "global"
    
    @register_analysis("predicted_vs_real")
    def predicted_vs_real(self):
        print("hello from pvr")
    
    predicted_vs_real.execution="running"

    @register_analysis("R2")
    def R2(self):
        print("Hello from R2")
    R2.execution="global"

    @register_analysis("histogram_error")
    def histogram_error(self):
        print("Hello from h_e")
    
    histogram_error.execution="global"

