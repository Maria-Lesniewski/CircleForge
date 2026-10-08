from ..diagnostics_registry import DiagnosticsRegistry
from .base import StatefulAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
from numpy import sqrt
import torch

@DiagnosticsRegistry.register_analysis("avg_and_dev")
class avg_and_dev(StatefulAnalysis):
    """
    Calculates the mean and standard deviation of each passed state
    stdev is calculated as sqrt of the variance. Both population and sample 
    variance estimates are reported. 
    (Generally sum(x - u)^2 = sum(x^2) - N*u^2 )
    (if pop var -> divide by N, if var estimate from sample-> divide by N -1)
    We also report the estimated error in the mean from each estimate type 
    """
    def __init__(self, keys : list[str] = None):
        super().__init__()
        self.state = {}
        for ele in keys:
            self.state[ele] = {
                    "sum_x" : 0,
                    "N" : 0,
                    "sum_x2" : 0,
                    }

    def running(self, streamer : StreamingState, scribe : Logger, inputs : list[str] = None):
        for ele in self.state:
            self.state[ele]["N"] += len(streamer.get(ele))
            self.state[ele]["sum_x"] += torch.sum((streamer.get(ele))).item()
            self.state[ele]["sum_x2"] += torch.sum((streamer.get(ele))**2).item()

    def final(self, scribe : Logger, inputs : list[str] = None):
        for ele in self.state:
            mean = self.state[ele]["sum_x"] / self.state[ele]["N"]
            N = self.state[ele]["N"]
            pop_stddev = sqrt( 1 / N *(self.state[ele]["sum_x2"] - mean**2*N))
            pop_SE = pop_stddev / sqrt(N)
            if N > 1:
                sample_stddev = sqrt( 1 / (N-1)*(self.state[ele]["sum_x2"] - mean**2*N))
                sample_SE = sample_stddev / sqrt(N)
            else:
                sample_stddev = "Not Available"
                sample_SE = "Not Available"
            data = {
                    "arithmatic_mean" : mean,
                    "population_stddev" : pop_stddev,
                    "sample_stddev" : sample_stddev,
                    "pop_SE_est" : pop_SE,
                    "sample_SE_est" : sample_SE
                    }

            scribe.write_json(f"{ele}_dist_stats.json",data)
