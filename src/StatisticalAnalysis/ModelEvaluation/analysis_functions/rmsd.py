from ..analysis_registry import AnalysisRegistry
from .base import StatefulAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
from torch import sqrt
import torch

@AnalysisRegistry.register_analysis("rmsd")
class RMSD(StatefulAnalysis):
    """
    Calculates the root mean square deviation of the model wrt. samples
    e.g. (E(model - true)^2)^(1/2)
    """
    def __init__(self):
        super().__init__()
        self.state = {
                "sum" : 0,
                "N" : 0,
                }

    def running(self, streamer : StreamingState, scribe : Logger = None):
        self.state["N"] += len(streamer.prediction_state)
        self.state["sum"] += torch.sum((streamer.prediction_state - streamer.label_state)**2)

    def final(self, recorder : Logger):
        recorder.add_scalars(
                "evaluation",
                {"RMSD" : sqrt(self.state["sum"]/self.state["N"])},
                0)
