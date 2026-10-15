from ..diagnostics_registry import DiagnosticsRegistry
from .base import ConfigurableAnalysis
from ...streaming_state import StreamingState
from utils.model_logger import Logger
import utils.io as io
from numpy import sqrt
import torch
import torch.nn.functional as F

@DiagnosticsRegistry.register_analysis("correlate_feature")
class correlate_feature(ConfigurableAnalysis):
    """
    Calculates correlations between features/manipulations of features and
    target labels 
    """
    def __init__(self, keys : list[str]):
        super().__init__()
        self.keys = keys
        self.state = {} # FILL OUT 
        self.config = {} # DURING CONFIG OBTAIN
        self.choice = 0

    def final(self, scribe : Logger):
        """
        Print metadata of transformation choices
        """
        scribe.write_json("feature_transform_settings.json", {"selected_functions" : list(self.config.keys())})
        

    def L2(self, feature : torch.Tensor) -> float:
        """
        Gets L2 norm of an image
        """
        return torch.linalg.norm(feature, ord=2).item()

    def mean(self, feature : torch.Tensor) -> float:
        """
        Returns mean pixel value of an image
        """
        return feature.mean().item()

    def LP(self, feature : torch.Tensor) -> float:
        """
        Returns the average of a 3x3 laplacian filter passed over the image
        """
        kernel = torch.tensor(
                [
                [ 0, -1,  0],
                [-1,  4, -1],
                [ 0, -1,  0],
                ],
                dtype=torch.float32, requires_grad=False)


        """ # Idea for larger edges. Didnt work
        torch.tensor(
                [
                [ 0,  0, -1, -1,  0,  0],
                [ 0, -1, -2, -2, -1,  0],
                [-1, -2,  4,  4, -2, -1],
                [-1, -2,  4,  4, -2, -1],
                [ 0, -1, -2, -2, -1,  0],
                [ 0,  0, -1, -1,  0,  0],
                ],
                dtype=torch.float32, requires_grad=False)
        """
        response = F.conv2d(
                feature[None, None], # We are passing one sample at a time.
                kernel[None, None], # but this function expects batch dimension and multiple signal channels
                )
        return response.abs().mean().item()


    _feature_pooling_funcs = { # We're not using a registry factory this time
            "mean" : mean,
            "L2Norm" : L2,
            "LaplacianResponse" : LP
            }

    def obtain_config(self):
        choice = io.compounding_multiple_choice_q(
                    extra_info=("'correlate_feature' requires a feature transform choice \n" +
                    "for pooling spacial data into a scalar\n"),
                    multiple_choice=list(self._feature_pooling_funcs.keys()))

        for ele in choice:
            self.config[ele] = self._feature_pooling_funcs[ele]
            self.state[ele] = []

    def running(self, streamer : StreamingState, scribe : Logger, inputs : list[str] = None):

        # Zero out previous running state analysis (batch data only)
        for choice in self.config.keys():
            self.state[choice] = []
        
        # Call transforms, note we instantiate them unbound so need to pass self
        for transform in self.config.keys():
            for ele in streamer.features:
                self.state[transform].append(
                    self.config[transform](self,ele)
                    )
        # Write results against each label
        for ele in self.keys:
            for choice in self.config.keys():
                for i in range(len(streamer.features)):
                    scribe.add_scalars(
                            f"{ele}_vs_{choice}_on_feature",
                            {f"{ele}" : streamer.get(ele)[i]},
                            self.state[choice][i])
