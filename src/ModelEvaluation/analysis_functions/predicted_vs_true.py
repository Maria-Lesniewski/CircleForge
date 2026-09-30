from ..analysis_registry import AnalysisRegistry
from .base import BasicAnalysis
from ..streaming_state import StreamingState
from utils.model_logger import Logger

@AnalysisRegistry.register_analysis("predicted_vs_true")
class predicted_vs_true(BasicAnalysis):
    """
    prints model prediction against actual data label
    """
    @classmethod
    def running(cls, stream : StreamingState, recorder : Logger):
        for i in range(len(stream.prediction_state)):
            recorder.add_scalars(
                "evaluation",
                {"predicted_vs_true" : stream.prediction_state[i]},
                stream.label_state[i]
                )
