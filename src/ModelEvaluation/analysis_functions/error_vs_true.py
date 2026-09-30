from ..analysis_registry import AnalysisRegistry
from .base import BasicAnalysis
from ..streaming_state import StreamingState
from utils.model_logger import Logger

@AnalysisRegistry.register_analysis("error_vs_true")
class error_vs_true(BasicAnalysis):
    """
    prints the error in model prediction
    against the true value
    """
    @classmethod
    def running(cls, stream : StreamingState, scribe : Logger):
        for i in range(len(stream.prediction_state)):
                scribe.add_scalars(
                "evaluation",
                {"error_vs_true" : stream.prediction_state[i] - stream.label_state[i]},
                stream.label_state[i]
                )
