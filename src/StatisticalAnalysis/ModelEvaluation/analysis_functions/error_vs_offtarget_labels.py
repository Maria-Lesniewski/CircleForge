from ..analysis_registry import AnalysisRegistry
from .base import BasicAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger

@AnalysisRegistry.register_analysis("error_vs_offtarget_labels", "special")
class error_vs_true(BasicAnalysis):
    """
    prints the error in model prediction
    against other labels within the data set
    """

    def __init__(self, keys : list[str]):
        super().__init__()
        self.keys = keys

    def running(self, stream : StreamingState, scribe : Logger):

        for ele in self.keys:
            for i in range(len(stream.prediction_state)):
                scribe.add_scalars(
                "evaluation",
                {f"error_vs_{ele}" : stream.prediction_state[i] - stream.label_state[i]},
                stream.get(ele)[i]
                )
