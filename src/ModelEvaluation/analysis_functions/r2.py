from ..analysis_registry import AnalysisRegistry
from .base import StatefulAnalysis
from ..streaming_state import StreamingState
from utils.model_logger import Logger
import torch 

@AnalysisRegistry.register_analysis("R2")
class R2(StatefulAnalysis):
    """
    Calculates R2 for the model wrt the current set (should be bounded btwn 0 and 1 for training set)
    e.g.
    R2 = 1 - SSres/SSTot
    SSres = sum(model - true)^2
    SSTot = sum(true - true_mean)^2 = sum(true^2) - sum(true)^2/N
    """

    state = {
        "SSE" : 0,
        "true_sum" : 0,
        "true_square" : 0,
        "N" : 0
        }
    @classmethod
    def running(cls, stream : StreamingState, scribe : Logger = None):
        cls.state["N"] += len(stream.prediction_state)
        cls.state["SSE"] += torch.sum((stream.prediction_state - stream.label_state)**2)
        cls.state["true_sum"] += torch.sum(stream.label_state)
        cls.state["true_square"] += torch.sum(stream.label_state**2)
    
    @classmethod
    def final(cls, scribe : Logger):
        SSTot = cls.state["true_square"] - cls.state["true_sum"]**2/cls.state["N"]
        var = SSTot / cls.state["N"]
        r2 = 1 - (cls.state["SSE"] / SSTot)
        scribe.add_scalars(
                "evaluation",
                {
                    "R2" : r2, 
                    "var" : var, 
                    "SSE" : cls.state["SSE"],
                    "SST" : SSTot, 
                    "N" : cls.state["N"]
                    },
                0
                )

        print(f"Final r2 {r2}")
        cls.state["SSE"] = 0
        cls.state["true_sum"] = 0
        cls.state["true_square"] = 0
        cls.state["N"] = 0
