from ..analysis_registry import AnalysisRegistry
from .base import ConfigurableAnalysis
from ...streaming_state import StreamingState
from torch import rand, histogram
from utils.io import dictionary_filling_question
from utils.model_logger import Logger
from utils.type_checker import is_float, is_int

@AnalysisRegistry.register_analysis("accuracy_within_tol")
class AccuracyWithinTol(ConfigurableAnalysis):

    def __init__(self):
        super().__init__()
        self.state = {
            "num_correct" : 0,
            "num_samples" : 0,
            }

        self.config = {
            "tol" : 0.01,
            }

    def running(self, stream : StreamingState, scribe : Logger):
        errors = stream.prediction_state - stream.label_state
        is_accurate = ([(abs(ele) < self.config["tol"]) for ele in errors])
        self.state["num_correct"] += sum(is_accurate).item()
        for i in range(len(is_accurate)):
            scribe.add_scalars(
                    "evaluation",
                    {"correct1_incorrect0" : is_accurate[i]},
                    self.state["num_samples"] + i
                    )
        self.state["num_samples"] += len(is_accurate)


    def final(self, scribe : Logger):
        config = self.config
        fin_state = self.config | self.state
        fin_state["final_%_accuracy"] = self.state["num_correct"]/self.state["num_samples"]*100
        scribe.write_json(dictionary=fin_state, rel_filepath="accuracy.json")

    def obtain_config(self):
        while True:
            user_config = dictionary_filling_question(
                    extra_info=("accuracy_within_tol requires a tolerance for what predictions count as 'correct' \n" +
                    "predictions with abs(error) < tol will be counted\n"),
                    keys=["tol"]
                    )
            if ( is_float(user_config["tol"])
                    ):
                self.config["tol"] = float(user_config["tol"])
                return
            else:
                print(f"Error in registered responses \n'{user_config}'\n")
                print("We expect floats for tolerance")
                print("Please try again")
