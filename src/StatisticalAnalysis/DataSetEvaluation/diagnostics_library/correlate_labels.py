from ..diagnostics_registry import DiagnosticsRegistry
from .base import BasicAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
from itertools import combinations

@DiagnosticsRegistry.register_analysis("correlate_labels")
class correlate_labels(BasicAnalysis):
    """
    prints pairs of streamed values against one another

    """

    def __init__(self, keys : list[str]):
        super().__init__()
        self.keys = keys
        self.running = self.on_running
        print("KEYS")
        print(self.keys)

        if len(self.keys) < 2:
            print("Warning: You've selected 'correlate_labels' for a data set that has 1 label")
            print("Accordingly we will skip this analysis and 'correlate_labels' will be empty")
            self.running = self.off_running


    # Psuedo strategy impls. 
    def off_running(self, stream : StreamingState, scribe : Logger, inputs : list[str] = None):
        # This function should do nothing. 
        pass

    def on_running(self, stream : StreamingState, scribe : Logger, inputs : list[str] = None):
        """
        Prints a running pairwise list of all values in streaming state as labeled by inputs
        """

        correlation_pairs = list(combinations(self.keys, 2))

        for x, y in correlation_pairs:
            for i in range(len(stream.get(x))):
                scribe.add_scalars(
                "diagnostic",
                {f"{y}_vs_{x}" : stream.get(y)[i]},
                stream.get(x)[i]
                )
