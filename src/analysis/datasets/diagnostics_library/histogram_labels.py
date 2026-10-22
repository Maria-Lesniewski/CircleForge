from ..diagnostics_registry import DiagnosticsRegistry
from .base import ConfigurableAnalysis
from ...streaming_state import StreamingState
from torch import rand, histogram
from utils.io import dictionary_filling_question
from utils.model_logger import Logger
from utils.type_checker import is_float, is_int

@DiagnosticsRegistry.register_analysis("histogram_labels")
class HistogramLabels(ConfigurableAnalysis):
    """
        Fills in a running histogram of (signed) error within bounds specified by user
        Notes any error over/underflow count and returns actual mins/max for repeated analysis

        Attributes:
            -state : Internal structure for calculation
                -hist : The count
                -edges : The bins
                -observed_min : The minimum label value
                -observed_max : The maximum label balue
                -overflow_count : The number of samples not included in the histogram at the high end
                -underflow_count : the number of samples truncated to the lowest bin 

            -config: Analysis config set by the user on request
                -max_bin : The highest label value to bin during the analysis
                -min_bin : The lowest label value to bin during the analysis
                -n_bins : The resolution of the histogram along domain [min_bin, max_bin)
    """

    def __init__(self, keys : list[str]):
        super().__init__()
        self.state = {}
        self.config = {}
        self.keys = keys
        # For now just one config for all, save custom for later rework to config files
        self.config = {
                "max_bin" : 0,
                "min_bin" : 0,
                "n_bins" : 0
                }
        for ele in keys:
            self.state[ele] = {
                    "hist" : 0,
                    "edges" : 0,
                    "observed_min" : float("inf"),
                    "observed_max" : float("-inf"),
                    "overflow_count" : 0,
                    "underflow_count" : 0
                    }


    def running(self, stream : StreamingState, scribe : Logger = None, keys : list[str] = None):
        for ele in self.keys:
            label_val = stream.get(ele)
            batch_hist, _ = histogram(label_val, bins=self.config["n_bins"], range=(self.config["min_bin"], self.config["max_bin"]))
            self.state[ele]["hist"] += batch_hist
            ovflo = [entry > self.config["max_bin"] for entry in label_val]
            unflo = [entry < self.config["min_bin"] for entry in label_val]
            self.state[ele]["overflow_count"] += sum(ovflo).item()
            self.state[ele]["underflow_count"] += sum(unflo).item()
            if any(entry > self.state[ele]["observed_max"] for entry in label_val):
                self.state[ele]["observed_max"] = max(label_val).item()
            if any(entry < self.state[ele]["observed_min"] for entry in label_val):
                self.state[ele]["observed_min"] = min(label_val).item()

    def final(self, scribe : Logger):
        for ele in self.keys:
            config = self.config
            fin_state = dict(list(self.state[ele].items())[2:])
            scribe.write_json(dictionary=config, rel_filepath="{ele}_histogram_config.json")
            scribe.write_json(dictionary=fin_state, rel_filepath=f"{ele}_histogram_meta.json")
            for binloc, count in zip(self.state[ele]["edges"], self.state[ele]["hist"]):
                scribe.add_scalars(
                        f"evaluation_{ele}",
                        {"histogram" : count},
                        binloc
                        )

    def obtain_config(self):
        # For now just one config for the whole thing 
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
                for ele in self.keys:
                    _, self.state[ele]["edges"] = histogram(
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
