# @file ModelAnalyzer.py
# @author Maria Lesniewski
# @desc This is a class to abstract model analysis into one callable interface and modular functions to test. 
# @desc Generally, all metrics will require model to evaluate data. Rather than doing that
#       repeatedly, this class registers each modular analysis type and reuses the same predictions
#       as requested by a user to log data / performance
from typing import Dict, Any
import torch
import torch.nn as nn
from utils.model_logger import Logger
import utils.io as io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
    def register_analysis(analysis_name: str, analysis_availability : str, registry=_function_registry):
        def decorator(analysis_function):
            """
            decorator factory for adding to dictionary of implemented analysis routines
            """
            analysis_function.available = analysis_availability
            registry[analysis_name] = analysis_function
            return analysis_function
        return decorator

    def __init__(self, model : nn.Module , dataset : Dataset, filewriter_root_path : str, evaluate_against : str, batchsize : int = 32):
        """
        Obj Attributes:

            model : nn.Module - Usually loaded from a .cpt then passed to this init
            data_set_loader : DataLoader - Set for which evaluation should happen
            evaluation_objective : str - The name of the target for which the model should be evaluated against. (i.e. what does the model predict) 
            filewriter_root_path : str - root path that all analysis will be saved under in sub folders
            evaluate_against : str - The ground truth label the model is trying to predict
        """
        self.model : nn.Module = model
        self.evaluate_against = evaluate_against
        self.dataloader : DataLoader = DataLoader(dataset, batchsize, shuffle=False, num_workers=1)
        self.datadict : dict[str, int] = {} 
        self.init_loader_dict()
        self.filewriter_root_path : str = filewriter_root_path
        self.schedule : Dict[str,Any] = {}
        self.check_evaluation_request()
        self.prediction_state = 0 # This will store the data the model predicts batchwise
        self.label_state = 0 # This will store the true values of the target objective.
        self.loggers : dict[str, Logger] = {}

    def init_loader_dict(self):
        """
        Obtains the column names of data from the loaded dd dataset
        to make indexing specific targets easier later
        """
        for i, ele in enumerate(self.dataloader.dataset.target_columns):
            self.datadict[ele] = i

    def check_evaluation_request(self):
        """
        Verifies data loader has required target within it
        """

        if (self.evaluate_against not in self.dataloader.dataset.target_columns):
            error_str = (f"target_objective '{self.evaluate_against}' not in data loader\n" +
                    f"Received dataset with columns {self.dataloader.target_columns}")
            raise ValueError(error_str)

    def obtain_schedule(self):
        """
        Prints list of all possible implemented analysis, 
        if analysis has options, asks for relevant options 
        """

        # Obtain user calc req
        intro_context = ("The Model Analyzer is used to evaluate pre-trained models\n" +
                "The following are currently implemented analysis routines\n")
        choices = io.compounding_multiple_choice_q(intro_context, list(self._function_registry.keys()))
        # register calc reqs to schedule, create working dir for analysis and loggers
        a_path = io.make_artifact_dir(f"Analysis_{self.model.__class__.__name__}")
        print(f"saving analysis in {a_path}")
        for ele in choices:
            self.schedule[ele] = self._function_registry[ele]
            self.loggers[ele] = Logger(a_path / ele)

    def run_schedule(self):
        """
        Perform and log each calculation listed
        """
        # Iterate over the data to accumulate stats for requests
        # functions that need no further work write results within
        # functions that need global data will update "states" of the function 
        for i, data in enumerate(self.dataloader):
            input_img, label = data
            self.prediction_state = self.model(input_img)
            self.label_state = label[:,self.datadict[self.evaluate_against]]
            mode="debugging"
            for func in self.schedule.keys():
                self.schedule[func](self, mode)



#### Begin Grabhat of Function Implementations 
    @register_analysis("RMSD", "always")
    def RMSD(self, mode):
        """
        Calculates the root mean square deviation of the model wrt. samples
        e.g. (E(model - true)^2)^(1/2)
        """
        if mode == "running":
            RMSD.state["N"] += len(data)
            RMSD.state["sum"] += ()**2
        if mode == "final":
            print("Hello final mode")
            # write sqrt(sum)/N somewhere
    RMSD.state = {
            "sum" : 0,
            "N" : 0, 
            }
    
    @register_analysis("predicted_vs_real", "always")
    def predicted_vs_real(self, mode=None):
        # print model prediction against actual data
        for i in range(len(self.prediction_state)):
            print(f"pred: {self.prediction_state[i]}")
            print(f"lab: {self.label_state[i]}")
            self.loggers["predicted_vs_real"].add_scalars(
                    "evaluation", 
                    {"predicted_vs_real" : self.prediction_state[i].item()},
                    self.label_state[i]
                    )
    @register_analysis("R2", "always")
    def R2(self, mode):
        print("Hello from R2")

    @register_analysis("histogram_error", "always")
    def histogram_error(self, mode):
        print("Hello from h_e")

    @register_analysis("RMSE_by_category", "many_target")
    def RMSE_by_category(self, mode):
        print("Hello from RMSE_by_category")

    def _reset_functional_state(func):
        """
        for functions with persistant state attributes, 
        resets state attribute to zero for next calculation
        """
        for item in func.state.keys():
            func.state[item] = 0
