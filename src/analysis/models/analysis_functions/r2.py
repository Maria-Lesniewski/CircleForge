from ..analysis_registry import AnalysisRegistry
from .base import StatefulAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
import torch 


@AnalysisRegistry.register_analysis("r2")
class R2(StatefulAnalysis):
    """
    Calculates R2 for the model wrt the current set (should be bounded btwn 0 and 1 (or numerically very close) for training sets with least squares difs

    Calculates via Welford's alg. 
    https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    """
    def __init__(self):
        super().__init__()
        self.state = {
            "SSE": 0.0,
            "N": 0,
            "mean": 0.0, # running mean of targets
            "M2": 0.0, # Squared deviations from the last mean
        }

    def running(self, stream: StreamingState, scribe: Logger = None):
        predictions = stream.prediction_state.reshape(-1).double()
        targets = stream.label_state.reshape(-1).double()

        # Residual sum of squares
        self.state["SSE"] += torch.sum(
            (predictions - targets) ** 2
        ).item()

        # Update running mean and sum of squared deviations
        for value in targets:
            value = value.item()
            n = self.state["N"] + 1
            delta = value - self.state["mean"]
            self.state["mean"] += delta / n
            delta2 = value - self.state["mean"]
            self.state["M2"] += delta * delta2
            self.state["N"] = n

    def final(self, scribe: Logger):
        sstot = self.state["M2"]

        if sstot > 0:
            r2 = 1.0 - self.state["SSE"] / sstot
        else:
            r2 = float("nan")

        scribe.add_scalars(
            "evaluation",
            {
                "R2": r2,
                "SSE": self.state["SSE"],
                "SST": sstot,
                "N": self.state["N"],
                "var": sstot / self.state["N"]
            },
            0,
        )

        print(f"Final r2 {r2}")

