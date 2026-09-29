import DataSet.DoodleData as dd
import torch
import torch.nn as nn
import ModelEvaluation.model_analyzer as ma
import Models.model_options as mo
import utils.io as io
from pathlib import Path


### SKELETON 
# load model
# load data
# ma.obtain_config
# ma.run_configs


## Eventually we should recognize models saved from logger and just ask which to eval (Build a model config grabber from the models stored by loggers)
config = {"model_type" : "BasicLinear", "model_path" : "tests/ModelTests/modelstates/BasicLinear.pt", "evaluation_objective" : "ncircles" }
## For now we'll hard code config and use our least squares model as a test

# Load Model
model = mo.ModelFactory.create(config["model_type"], {})
model.load_state_dict(torch.load(Path(io.find_project_root() / config["model_path"])))
model.eval()

# Load Data Sets
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=[
            "ncircles",
            "nbig"
            ],
        split_tag="train"
        )
val_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="val"
        )
test_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="test"
        )

analyzer = ma.ModelAnalyzer(model, train_data, "./", config["evaluation_objective"])
analyzer.obtain_schedule()
analyzer.run_schedule()
