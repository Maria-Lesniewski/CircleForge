import utils.io as io
import sys
from launcher.experiment_runner import run_job

scripts_directory = io.find_project_root() / "scripts"
here = io.find_here()

# Least Squares Training 
training_CL_args = [str(scripts_directory / "direct_linear_model_solver.py"), "--config", f"configs/least_squares_config.py", "--model", "BasicLinear"]
training_cmd = [sys.executable, *training_CL_args]
run_job(training_cmd, inp=None)

training_CL_args = [str(scripts_directory / "direct_linear_model_solver.py"), "--config", f"configs/least_squares_config.py", "--model", "RotatedLinear"]
training_cmd = [sys.executable, *training_CL_args]
run_job(training_cmd, inp=None)

training_CL_args = [str(scripts_directory / "direct_linear_model_solver.py"), "--config", f"configs/least_squares_config.py", "--model", "Constant"]
training_cmd = [sys.executable, *training_CL_args]
run_job(training_cmd, inp=None)
