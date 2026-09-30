from ..analysis_registry import AnalysisRegistry

@AnalysisRegistry.register_analysis("histogram_error")
def histogram_error(self, mode):
    n_bins = self.histogram_error.__func__.config["n_bins"]
    minb = self.histogram_error.__func__.config["min_bin"]
    maxb = self.histogram_error.__func__.config["max_bin"]

    if mode == "running":
        errors = self.prediction_state - self.label_state
        batch_hist, _ = torch.histogram(errors, bins=n_bins, range=(minb, maxb))
        self.histogram_error.__func__.state["hist"] += batch_hist
        ovflo = [entry > self.histogram_error.__func__.config["max_bin"] for entry in errors]
        unflo = [entry < self.histogram_error.__func__.config["min_bin"] for entry in errors]
        self.histogram_error.__func__.state["overflow_count"] += sum(ovflo).item()
        self.histogram_error.__func__.state["underflow_count"] += sum(unflo).item()
        if any(entry > self.histogram_error.__func__.state["observed_max"] for entry in errors):
            self.histogram_error.__func__.state["observed_max"] = max(errors).item()
        if any(entry < self.histogram_error.__func__.state["observed_min"] for entry in errors):
            self.histogram_error.__func__.state["observed_min"] = min(errors).item()

    if mode == "final":
        config = self.histogram_error.__func__.config
        for key, value in config.items():
            print(key, type(value), value)
        fin_state = dict(list(self.histogram_error.__func__.state.items())[2:])
        self.loggers["histogram_error"].write_json(dictionary=config, rel_filepath="histogram_config.json")
        self.loggers["histogram_error"].write_json(dictionary=fin_state, rel_filepath="histogram_meta.json")
        for binloc, count in zip(self.histogram_error.__func__.state["edges"], self.histogram_error.__func__.state["hist"]):
            self.loggers["histogram_error"].add_scalars(
                    "evaluation",
                    {"histogram" : count},
                    binloc
                    )

histogram_error.state = {
        "hist" : 0,
        "edges" : 0,
        "observed_min" : float("inf"),
        "observed_max" : float("-inf"),
        "overflow_count" : 0,
        "underflow_count" : 0
        }
histogram_error.config = {
        "max_bin" : 0,
        "min_bin" : 0,
        "n_bins" : 0
        }
def obtain_histogram_error_config(self):
    
    while True:
        user_config = io.dictionary_filling_question(
                extra_info="the histogram_error function requires additional configuration",
                keys=["max_bin", "min_bin", "n_bins"]
                )
        if ( tc.is_float(user_config["max_bin"])
                and tc.is_float(user_config["min_bin"])
                and tc.is_int(user_config["n_bins"])
                ):
            max_bin = float(user_config["max_bin"])
            min_bin = float(user_config["min_bin"])
            n_bins = int(user_config["n_bins"])

            self.histogram_error.__func__.config["max_bin"] = max_bin
            self.histogram_error.__func__.config["min_bin"] = min_bin
            self.histogram_error.__func__.config["n_bins"] = n_bins
            _ , self.histogram_error.__func__.state["edges"] = torch.histogram(torch.rand(100), range=(min_bin, max_bin), bins=n_bins)
            return
        else:
            print(f"Error in registered responses \n'{user_config}'\n")
            print("We expect floats for max_bin and min_bin")
            print("We expect an integer for n_bins")
            print("Please try again")

histogram_error.obtain_config = obtain_histogram_error_config

@AnalysisRegistry.register_analysis("RMSD_by_category", "special")
def RMSD_by_category(self, mode):
    print("Hello from RMSE_by_category")

def _reset_functional_state(func):
    """
    for functions with persistant state attributes, 
    resets state attribute to zero for next calculation
    """
    for item in func.__func__.state.keys():
        func.state[item] = 0

