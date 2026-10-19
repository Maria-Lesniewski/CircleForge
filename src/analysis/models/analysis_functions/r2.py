from ..analysis_registry import AnalysisRegistry
from .base import StatefulAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
import torch 

@AnalysisRegistry.register_analysis("r2")
class R2(StatefulAnalysis):
    """
    Calculates R2 for the model wrt the current set (should be bounded btwn 0 and 1 for training set)
    e.g.
    R2 = 1 - SSres/SSTot
    SSres = sum(model - true)^2
    SSTot = sum(true - true_mean)^2 = sum(true^2) - sum(true)^2/N
    """

    def __init__(self):
        super().__init__()
        self.state = {
            "SSE" : 0,
            "true_sum" : 0,
            "true_square" : 0,
            "N" : 0
            }
        
    def running(self, stream : StreamingState, scribe : Logger = None):
        self.state["N"] += len(stream.prediction_state)
        self.state["SSE"] += torch.sum((stream.prediction_state - stream.label_state)**2)
        self.state["true_sum"] += torch.sum(stream.label_state)
        self.state["true_square"] += torch.sum(stream.label_state**2)
    
    def final(self, scribe : Logger):
        SSTot = self.state["true_square"] - self.state["true_sum"]**2/self.state["N"]
        var = SSTot / self.state["N"]
        if SSTot != 0:
            r2 = 1 - (self.state["SSE"] / SSTot)
        else:
            r2 = float("nan")
        scribe.add_scalars(
                "evaluation",
                {
                    "R2" : r2, 
                    "var" : var, 
                    "SSE" : self.state["SSE"],
                    "SST" : SSTot, 
                    "N" : self.state["N"]
                    },
                0
                )

        print(f"Final r2 {r2}")
