# @file analysis_runner.py
# @author Maria Lesniewski
# @desc This is a class to abstract model analysis into one callable interface and modular functions to test. 
# @desc Generally, all metrics will require model to evaluate data. Rather than doing that
#       repeatedly, this class registers each modular analysis type and reuses the same predictions
#       as requested by a user to log data / performance

from typing import Any
import torch
import torch.nn as nn
import utils.type_checker as tc
from utils.model_logger import Logger
from .diagnostics_registry import DiagnosticsRegistry
from ..streaming_state import StreamingState
import utils.io as io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
class DiagnosticsRunner:
    """
    Coordinates modular analysis routines for a dataset

    Attributes:
            dataloader - iterator over passed data set used for evaluation
            datadict - index map for when dataset contains multiple off target labels
            filewriter_root_path - root path for saving all analysis (subfolders created dependent on analysis)
            schedule - list of analysis routines to be executed
            streamer - streaming object holding the feature and label states while running through dataset
            loggers - writing objects for recording each of the scheduled analyses
    """
    def __init__(self, dataset : Dataset, filewriter_root_path : str, batchsize : int = 32):
    
        self.dataloader : DataLoader = DataLoader(dataset, batchsize, shuffle=False, num_workers=1)
        self.datadict : dict[str, int] = {} 
        self.init_loader_dict()
        self.filewriter_root_path : str = filewriter_root_path
        self.schedule : dict[str,Any] = {} # Holds instances of analysis functions with analysis state variables
        self.schedule_options : dict[str,Any] = {}
        self.streamer = 0
        self.init_streamer()
        self.loggers : dict[str, Logger] = {}

    def init_loader_dict(self):
        """
        Obtains the column names of data from the loaded dd dataset
        to make indexing specific targets easier later
        """
        for i, ele in enumerate(self.dataloader.dataset.target_columns):
            self.datadict[ele] = i

    def init_streamer(self):
        """
        Uses the column names from the loaded dd dataset to determine 
        the state labels that will be streamed
        """
        unified_dict = {}
        unified_dict = {"features" : 0}
        for ele in self.dataloader.dataset.target_columns:
            unified_dict[ele] = 0

        self.streamer = StreamingState(unified_dict)

    def obtain_schedule(self):
        """
        Prints list of all possible implemented analysis, 
        if analysis has options, asks for relevant options 
        """

        # Obtain user calc req
        intro_context = ("The Diagnostics Runner is used to analyze patterns existent in feature/label data\n" +
                "The following are currently implemented analysis routines\n")

        # We can relate performance to off target labels if they're available.
        # if not standard options apply
        options = DiagnosticsRegistry.available()
        self.schedule_options = options

        choices = sorted(io.compounding_multiple_choice_q(intro_context, list(options.keys())))
        # register calc reqs to schedule, create working dir for analysis and loggers
        a_path = Path(self.filewriter_root_path)
        a_sub_path = a_path / f"{self.dataloader.dataset.split_tag}"

        a_sub_path.mkdir(parents=True, exist_ok=True)

        for ele in choices:
            self.schedule[ele] = DiagnosticsRegistry.create(ele, {"keys" : list(self.datadict.keys())})
            if hasattr(self.schedule[ele], "config"):
                self.schedule[ele].obtain_config()

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
            self.streamer.features = input_img
            # Add multi-labels to streamer
            for ele in self.datadict:
                self.streamer.assign(ele, label[:, self.datadict[ele]])

            # Do Analysis
            for func in self.schedule.keys():
                self.schedule[func].running(self.streamer, self.loggers[func], list(self.datadict.keys()))

        # Finish any standing state functions
        print(f"Finished iterating over {len(self.dataloader)} batches of {self.dataloader.batch_size} samples")
        print(f"Finalizing cumulative analyses")
        for func in self.schedule.keys():
            if hasattr(self.schedule[func], "state"):
                self.schedule[func].final(self.loggers[func])

    def close(self):
        for logger in self.loggers:
            self.loggers[logger].close()


