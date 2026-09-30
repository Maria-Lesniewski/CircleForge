from ..analysis_registry import AnalysisRegistry
from .base import StatefulAnalysis
from ..streaming_state import StreamingState
from utils.model_logger import Logger
from numpy import sqrt
import torch

@AnalysisRegistry.register_analysis("RMSD")
class RMSD(StatefulAnalysis):
    """
    Calculates the root mean square deviation of the model wrt. samples
    e.g. (E(model - true)^2)^(1/2)
    """
    state = {
            "sum" : 0,
            "N" : 0,
            }

    @classmethod
    def running(cls, streamer : StreamingState, scribe : Logger = None):
        cls.state["N"] += len(streamer.prediction_state)
        cls.state["sum"] += torch.sum((streamer.prediction_state - streamer.label_state)**2)

    @classmethod
    def final(cls, recorder : Logger):
        recorder.add_scalars(
                "evaluation",
                {"RMSD" : sqrt(cls.state["sum"]/cls.state["N"])},
                0)

        cls.state["sum"] = 0
        cls.state["N"] = 0
