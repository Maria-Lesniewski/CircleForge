import utils.io as io
import sys
from launcher.experiment_runner import run_job

scripts_directory = io.find_project_root() / "scripts"
here = io.find_here()

# Cores for parallel portion
NUM_PROCS = 4 # <---- Adjust this for your system

# Generator Run
generator_CL_args = [str(scripts_directory / "generate_training_data2.py"), "--config" ,"configs/generator_config.py"]
# ---------------------------------------- Leave blank ^ for default settings

generator_cmd = ["mpirun", "-np", str(NUM_PROCS), sys.executable, *generator_CL_args]
run_job(generator_cmd, inp="Y\nY\n") # Generator asks "are you sure" up to twice before activating.

# Data Split + Remove Duplicates
split_CL_args = [str(scripts_directory / "scrub_and_split_data.py"), "--config","configs/data_split_config.py"]
# ---------------------------------------------------------------- ^ blank for default settings
# ---------------------------------------------------------------but function depends on the generator
split_command = [sys.executable, *split_CL_args]
run_job(split_command)

# Data Analysis. This is a good point to stop and look manually at your data (hence multiple part pipeline in experiment runners)
# Data Analysis Options, currently a menu comes up with 4 analysis (0-3) that need to be configured on the CL
CL_string = ""
for i in range(0,3): # for each dataset analyzer
    CL_string+="0 1 2 3 -1\n" # Pick all options
    CL_string+="0 1 2 -1\n"
    CL_string+="200\n"# Configure the histogram - max_bin
    CL_string+="0\n" # Configure the label histogram - min bin
    CL_string+="200\n" # configure the histogram - nbins
data_CL_args = [str(scripts_directory / "analyze_data_splits2.py"), "--config", "configs/data_analysis_config.py"]
data_analysis_command = [sys.executable, *data_CL_args]
run_job(data_analysis_command, inp=CL_string)
