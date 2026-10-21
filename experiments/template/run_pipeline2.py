import utils.io as io
import sys
from launcher.experiment_runner import run_job

# This is a pipeline template for after the data has been generated and any feature engineering complete

scripts_directory = io.find_project_root() / "scripts"
here = io.find_here()

# Training

if not (len(sys.argv) > 1):
    ### (New Model)
    training_CL_args = [str(scripts_directory / "train_models_config_based.py"), "--config","training_config.py"]
    training_cmd = [sys.executable, *training_CL_args] 
    run_job(training_cmd, inp="0\n") # train model asks for what kind of model when no --model is passed
    #   Current 0. BasicLinear 1. RotatedLinear 2. Constant 3. ConvRotatedLinear 
else:
    ### (Continuation w. presupplied model) (pass path to this script)
    training_CL_args = [str(scripts_directory / "train_models_config_based.py"), "--config","training_config.py", "--model", (sys.argv[1])]
    training_cmd = [sys.executable, *training_CL_args]
    run_job(training_cmd, inp=None)
