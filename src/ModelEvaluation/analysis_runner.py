# @file analysis_runner.py
# @author Maria Lesniewski
# @desc This is a class to abstract model analysis into one callable interface and modular functions to test. 
# @desc Generally, all metrics will require model to evaluate data. Rather than doing that
#       repeatedly, this class registers each modular analysis type and reuses the same predictions
#       as requested by a user to log data / performance

from typing import Dict, Any
import torch
import torch.nn as nn
import utils.type_checker as tc
from utils.model_logger import Logger
from .analysis_registry import AnalysisRegistry
from .streaming_state import StreamingState
import utils.io as io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class AnalysisRunner:
    """
    Coordinates modular analysis routines for a trained model.
    Designed to cut down on wasted time from multiple model inference passes

    Attributes:
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
    def __init__(self, model : nn.Module , dataset : Dataset, filewriter_root_path : str, evaluate_against : str, batchsize : int = 32):
        
        self.model : nn.Module = model
        self.evaluate_against = evaluate_against
        self.dataloader : DataLoader = DataLoader(dataset, batchsize, shuffle=False, num_workers=1)
        self.datadict : dict[str, int] = {} 
        self.init_loader_dict()
        self.filewriter_root_path : str = filewriter_root_path
        self.schedule : Dict[str,Any] = {}
        self.schedule_options : Dict[str,Any] = {}
        self.check_evaluation_request()
        self.streamer = StreamingState({"prediction_state" : 0, "label_state" : 0}) 
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

        # We can relate performance to off target labels if they're available.
        # if not standard options apply
        if len(self.datadict) > 1:
            options = AnalysisRegistry.total_available()
            self.schedule_options = options
        else:
            options = AnalysisRegistry.available()
            self.schedule_options = options

        choices = io.compounding_multiple_choice_q(intro_context, list(options.keys()))
        # register calc reqs to schedule, create working dir for analysis and loggers
        a_path = io.make_artifact_dir(f"Analysis_{self.model.__class__.__name__}")
        a_sub_path = io.make_artifact_dir(a_path / f"{self.dataloader.dataset.split_tag}")
        for ele in choices:
            self.schedule[ele] = self.schedule_options[ele]
            if hasattr(self.schedule_options[ele], "config"):
                self.schedule_options[ele].obtain_config()
            self.loggers[ele] = Logger(a_sub_path / ele)
        print(f"saving analysis in {a_sub_path}")


    def run_schedule(self):
        """
        Perform and log each calculation listed
        """
        # Iterate over the data to accumulate stats for requests
        # functions that need no further work write results within
        # functions that need global data will update "states" of the function 
        for i, data in enumerate(self.dataloader):
            input_img, label = data
            self.streamer.prediction_state = self.model(input_img).squeeze(-1)
            self.streamer.label_state = label[:,self.datadict[self.evaluate_against]]
            for func in self.schedule.keys():
                self.schedule[func].running(self.streamer, self.loggers[func])

        # Finish any standing state functions
        print(f"Finished iterating over {len(self.dataloader)} batches of {self.dataloader.batch_size} samples")
        print(f"Finalizing cumulative analyses")
        for func in self.schedule.keys():
            if hasattr(self.schedule[func], "state"):
                self.schedule[func].final(self.loggers[func])
