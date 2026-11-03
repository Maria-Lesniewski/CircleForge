import utils.io as io
import sys
from launcher.experiment_runner import run_job

# This is a pipeline template for after Models have been trained an now evaluation is required post training

scripts_directory = io.find_project_root() / "scripts"
here = io.find_here()

# Model Evaluation
# Model Evaluation Options, currently a menu comes up with:
    # case 1 (1 target_column in data): 6 analysis options (0-5)
    # case 2 (multiple target columns in data): 7 analysis options (0-6)
    # analyses usually need further config on the command line
CL_string = ""
for i in range(0,3): # for each dataset analyzer
    CL_string+="0 4 5 -1\n" # 0: accuracy, 4:r2, 5: rmsd 
    CL_string+="1\n" # Configure accuracy tolerance

for ele1 in ["a","b","c","d"]:
    for ele2 in ["a","b","c", "d"]:
        eval_CL_args = [str(scripts_directory / "analyze_models.py"), "--config",f"configs/model_evaluation_config_{ele1}_on_{ele2}.py"]
        eval_cmd = [sys.executable, *eval_CL_args] 
        run_job(eval_cmd, inp=CL_string) # train model asks for what kind of model when no --model is passed
