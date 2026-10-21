import utils.io as io
import sys
from launcher.experiment_runner import run_job

scripts_directory = io.find_project_root() / "scripts"
here = io.find_here()

# Cores for parallel portion
NUM_PROCS = 4 # <---- Adjust this for your system

# Generator Run
generator_CL_args = [str(scripts_directory / "generate_training_data2.py"), "--config" ,"generator_config.py"]
# ---------------------------------------- Leave blank ^ for default settings

generator_cmd = ["mpirun", "-np", str(NUM_PROCS), sys.executable, *generator_CL_args]
run_job(generator_cmd, inp="Y\nY\n") # Generator asks "are you sure" up to twice before activating.

# Data Split + Remove Duplicates
split_CL_args = [str(scripts_directory / "scrub_and_split_data.py"), "--config","data_split_config.py"]
# ---------------------------------------------------------------- ^ blank for default settings
# ---------------------------------------------------------------but function depends on the generator
split_command = [sys.executable, *split_CL_args]
run_job(split_command)


# Data Analysis. This is a good point to stop and look manually at your data (hence two experiment runners)
