# This is a file for holding the custom model_evaluation configuration
# (for configuring scripts/analyze_models.py)
# paths are relative to the artifact directories 
import glob
import re
import utils.io as io
cpt_path = io.artifact_path("nn_experiment/train_stats/ConvRotatedLinearReLU_best_checkpoint_epochs_training/cpt_meta_data")
files = glob.glob(str(cpt_path / "*.json"))
files = [f for f in files if "END_OF_ROUTINE" not in f]
highest = max(files, key=lambda f: int(re.search(r"_step_(\d+)", f).group(1)))

CONFIG = {
        "model" : {
            "type" : "ConvRotatedLinearReLU", # ModelFactoryKey string or "Ask" for prompt
            "meta_path" : str(f"{highest}"), # meta_data written by Trainer class. or "Ask" for prompt (generally put your last latest which will vary dependent on data gen process)
            },
        "data" : {
            "annotations_path" : "feature_engineering_experiment/final_data/" + "train_val_test_split_annotations.csv",
            "target_columns" : ["ncircles", "nbig", "nsmall"],
            "feature_transform" : lambda x: x/255
            },
        "analysis_settings" : {
            "evaluation_objective" : "ncircles", # str, colname
            "log_path" : "nn_experiment/ModelEvaluation", # Results stored under modename in outside folder
            } # Currently the schedule must be piped in via the CL interface
        }
