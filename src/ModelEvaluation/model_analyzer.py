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

            model - pre-trained inference machine passed in constructor
            evaluate_against - name of target for inference (i.e. what does the model predict)
            dataloader - iterator over passed data set used for evaluation
            datadict - index map for when dataset contains multiple off target labels
            filewriter_root_path - root path for saving all analysis (subfolders created dependent on analysis)
            schedule - list of analysis routines to be executed
            prediction_state - iterator holder for model predictions
            label_state - iterator holder for ground truths
            loggers - writing objects for recording each of the scheduled analyses
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
        # We can bin analysis by off target labels if they're available. If not, standard options

        choices = io.compounding_multiple_choice_q(intro_context, list(self._function_registry.keys()))
        # register calc reqs to schedule, create working dir for analysis and loggers
        a_path = io.make_artifact_dir(f"Analysis_{self.model.__class__.__name__}")
        a_sub_path = io.make_artifact_dir(a_path / f"{self.dataloader.dataset.split_tag}")
        print(f"saving analysis in {a_sub_path}")
        for ele in choices:
            self.schedule[ele] = self._function_registry[ele]
            self.loggers[ele] = Logger(a_sub_path / ele)

    def run_schedule(self):
        """
        Perform and log each calculation listed
        """
        # Iterate over the data to accumulate stats for requests
        # functions that need no further work write results within
        # functions that need global data will update "states" of the function 
        for i, data in enumerate(self.dataloader):
            input_img, label = data
            self.prediction_state = self.model(input_img).squeeze(-1)
            self.label_state = label[:,self.datadict[self.evaluate_against]]
            mode="running"
            for func in self.schedule.keys():
                self.schedule[func](self, mode)

        # Finish any standing state functions
        print(f"Finished iterating over {len(self.dataloader)} batches of {self.dataloader.batch_size} samples")
        print(f"Finalizing cumulative analyses")
        mode = "final"
        for fun in self.schedule.keys():
            if (self.schedule[func].state):
                self.schedule[func](self,mode)


#### Begin Grabhat of Function Implementations 
    @register_analysis("RMSD", "always")
    def RMSD(self, mode):
        """
        Calculates the root mean square deviation of the model wrt. samples
        e.g. (E(model - true)^2)^(1/2)
        """
        if mode == "running":
            self.RMSD.__func__.state["N"] += len(self.prediction_state)
            self.RMSD.__func__.state["sum"] += torch.sum((self.prediction_state - self.label_state)**2)
        if mode == "final":
            self.loggers["RMSD"].add_scalar(
                    "evaluation", 
                    {"RMSD" : np.sqrt(self.RMSD.__func__.state["sum"]/
                        self.RMSD.__func__.state["N"])
                        },
                    0)
            self._reset_functional_state(self.RMSD)

    RMSD.state = {
            "sum" : 0,
            "N" : 0, 
            }
    
    @register_analysis("predicted_vs_true", "always")
    def predicted_vs_true(self, mode=None):
        """
        prints model prediction against actual data label
        """
        for i in range(len(self.prediction_state)):
            self.loggers["predicted_vs_true"].add_scalars(
                    "evaluation", 
                    {"predicted_vs_true" : self.prediction_state[i]},
                    self.label_state[i]
                    )

    @register_analysis("R2", "always")
    def R2(self, mode):
        """
        Calculated R2 for the model wrt the current set (should be bounded btwn 0 and 1 for training set)
        e.g. 
        R2 = 1 - SSres/SSTot 
        SSres = sum(model - true)^2
        SSTot = sum(true - true_mean)^2 = sum(true^2) - sum(true)^2/N
        """
        if mode == "running":
            self.R2.__func__.state["N"] += len(self.prediction_state)
            self.R2.__func__.state["SSE"] += torch.sum((self.prediction_state - self.label_state)**2)
            self.R2.__func__.state["true_sum"] += torch.sum(self.label_state)
            self.R2.__func__.state["true_square"] += torch.sum(self.label_state**2)
        if mode == "final":
            SSTot = self.R2.__func__.state["true_square"] -\
                    self.R2.__func__.state["true_sum"]**2/self.R2.__func__.state["N"]
            var = SSTot / self.R2.__func__.state["N"]
            r2 = 1 - (self.R2.__func__.state["SSE"] / SSTot)
            self.loggers["R2"].add_scalars(
                    "evaluation",
                    {"R2" : r2, "var" : var, "SSE" : self.R2.__func__.state["SSE"], "SST" : SSTot, "N" : self.R2.__func__.state["N"]},
                    0
                    )

            print(f"Final r2 {r2}")

    R2.state = {
            "SSE" : 0, 
            "true_sum" : 0,
            "true_square" : 0,
            "N" : 0
            }

    @register_analysis("error_vs_true", "always")
    def error_vs_true(self, mode=None):
        """
        prints the error in model prediction 
        against the true value 
        """
        for i in range(len(self.prediction_state)):
            self.loggers["error_vs_true"].add_scalars(
                    "evaluation",
                    {"error_vs_true" : self.prediction_state[i] - self.label_state[i]},
                    self.label_state[i]
                    )

    @register_analysis("histogram_error", "always")
    def histogram_error(self, mode):
        print("Hello from h_e")

    @register_analysis("RMSD_by_category", "many_target")
    def RMSD_by_category(self, mode):
        print("Hello from RMSE_by_category")

    def _reset_functional_state(func):
        """
        for functions with persistant state attributes, 
        resets state attribute to zero for next calculation
        """
        for item in func.__func__.state.keys():
            func.state[item] = 0
