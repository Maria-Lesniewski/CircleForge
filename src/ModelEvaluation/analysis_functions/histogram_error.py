from ..analysis_registry import AnalysisRegistry
from .base import ConfigurableAnalysis
from ..streaming_state import StreamingState
from torch import rand, histogram
from utils.io import dictionary_filling_question
from utils.model_logger import Logger
from utils.type_checker import is_float, is_int

@AnalysisRegistry.register_analysis("histogram_error")
class histogram_error(ConfigurableAnalysis):

    def __init__(self):
        super().__init__()
        self.state = {
            "hist" : 0,
            "edges" : 0,
            "observed_min" : float("inf"),
            "observed_max" : float("-inf"),
            "overflow_count" : 0,
            "underflow_count" : 0
            }

        self.config = {
            "max_bin" : 0,
            "min_bin" : 0,
            "n_bins" : 0
            }

    def running(self, stream : StreamingState, scribe : Logger = None):
        errors = stream.prediction_state - stream.label_state
        batch_hist, _ = histogram(errors, bins=self.config["n_bins"], range=(self.config["min_bin"], self.config["max_bin"]))
        self.state["hist"] += batch_hist
        ovflo = [entry > self.config["max_bin"] for entry in errors]
        unflo = [entry < self.config["min_bin"] for entry in errors]
        self.state["overflow_count"] += sum(ovflo).item()
        self.state["underflow_count"] += sum(unflo).item()
        if any(entry > self.state["observed_max"] for entry in errors):
            self.state["observed_max"] = max(errors).item()
        if any(entry < self.state["observed_min"] for entry in errors):
            self.state["observed_min"] = min(errors).item()

    def final(self, scribe : Logger):
        config = self.config
        fin_state = dict(list(self.state.items())[2:])
        scribe.write_json(dictionary=config, rel_filepath="histogram_config.json")
        scribe.write_json(dictionary=fin_state, rel_filepath="histogram_meta.json")
        for binloc, count in zip(self.state["edges"], self.state["hist"]):
            scribe.add_scalars(
                    "evaluation",
                    {"histogram" : count},
                    binloc
                    )
        # close for repeated analysis
        self.state["hist"] = 0
        self.state["edges"] = 0
        self.state["observed_min"] = float("inf")
        self.state["observed_max"] = float("-inf")
        self.config["max_bin"] = 0
        self.config["max_bin"] = 0
        self.config["n_bins"] = 0

    def obtain_config(self):
        while True:
            user_config = dictionary_filling_question(
                    extra_info="the histogram_error function requires additional configuration",
                    keys=["max_bin", "min_bin", "n_bins"]
                    )
            if ( is_float(user_config["max_bin"])
                    and is_float(user_config["min_bin"])
                    and is_int(user_config["n_bins"])
                    ):
                max_bin = float(user_config["max_bin"])
                min_bin = float(user_config["min_bin"])
                n_bins = int(user_config["n_bins"])

                self.config["max_bin"] = max_bin
                self.config["min_bin"] = min_bin
                self.config["n_bins"] = n_bins
                _, self.state["edges"] = histogram(
                        rand(100), 
                        range=(min_bin, max_bin),
                        bins=n_bins
                        )
                return
            else:
                print(f"Error in registered responses \n'{user_config}'\n")
                print("We expect floats for max_bin and min_bin")
                print("We expect an integer for n_bins")
                print("Please try again")
