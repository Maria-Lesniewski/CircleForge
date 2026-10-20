from dataset.circle_data import CircleData
import torch
import torch.nn as nn
import analysis.models as ma
import models.model_options as mo
import utils.io as io
from pathlib import Path


# Config
config = {
        "BasicLinear" : "Results/checkpoints/BasicLinear_step_0.pt",
        "RotatedLinear" : "Results/checkpoints/RotatedLinear_step_0.pt",
        "Constant" : "Results/checkpoints/Constant_step_0.pt"
        }

# Load Models
models = {}
for ele in config:
    models[ele] = mo.ModelFactory.create(ele, {})
    models[ele].load_state_dict(torch.load(Path(io.find_here() / config[ele])))
    models[ele].eval()

# Load Data Sets
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = CircleData(
        csv_file=data_fp,
        target_columns=[
            "ncircles",
            ],
        split_tag="train"
        )
val_data = CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="val"
        )
test_data = CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="test"
        )

analysis_runners = {}
with torch.no_grad():
    for ele in models:
        analysis_runners[ele] = { 
                "train" : ma.AnalysisRunner(models[ele], train_data, "./", "ncircles"),
                "validation" : ma.AnalysisRunner(models[ele], val_data, "./", "ncircles"),
                "test" : ma.AnalysisRunner(models[ele], test_data, "./", "ncircles")
                }

        print(f"#### {ele} TRAINING SET EVALUATION SETTINGS ###")
        analysis_runners[ele]["train"].obtain_schedule()

        print(f"#### {ele} VALIDATION SET EVALUATION SETTINGS ###")
        analysis_runners[ele]["validation"].obtain_schedule()

        print(f"### {ele} TEST SET EVALUATION SETTINGS ###")
        analysis_runners[ele]["test"].obtain_schedule()

    for ele in models:
        print(f"Evaluating the {ele} model on different test sets")
        analysis_runners[ele]["train"].run_schedule()
        analysis_runners[ele]["validation"].run_schedule()
        analysis_runners[ele]["test"].run_schedule()

    for ele in models:
        analysis_runners[ele]["train"].close()
        analysis_runners[ele]["validation"].close()
        analysis_runners[ele]["test"].close()
