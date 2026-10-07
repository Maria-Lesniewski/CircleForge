from ..analysis_registry import AnalysisRegistry
from .base import BasicAnalysis
from ..streaming_state import StreamingState
from utils.model_logger import Logger

@AnalysisRegistry.register_analysis("correlation_plot")
class correlation_plot(BasicAnalysis):
    """
    prints the error in model prediction
    against the true value
    """

    def running(self, stream : StreamingState, scribe : Logger, inputs : ):
        for i in range(len(stream.prediction_state)):
                scribe.add_scalars(
                "diagnostic",
                {"error_vs_true" : stream.prediction_state[i] - stream.label_state[i]},
                stream.label_state[i]
                )
