from DataSet.circle_data import CircleData
import torch
import torch.nn as nn
import StatisticalAnalysis.ModelEvaluation as ma
import models.model_options as mo
import utils.io as io
from pathlib import Path


### SKELETON 
# load model
# load data
# ma.obtain_config
# ma.run_configs


## Eventually we should recognize models saved from logger and just ask which to eval (Build a model config grabber from the models stored by loggers)
#config = {"model_type" : "BasicLinear", "model_path" : "tests/ModelTests/modelstates/BasicLinear.pt", "evaluation_objective" : "ncircles" }
#config = {"model_type" : "BasicLinear", "model_path" : "tests/ModelTests/modelstates/Final_Least_Squares_opt_model", "evaluation_objective" : "ncircles" }
#config = {"model_type" : "RotatedLinear", "model_path" : "tests/ModelTests/modelstates/RotatedLinear.pt", "evaluation_objective" : "ncircles"}
config = {"model_type" : "RotatedLinear", "model_path" : "tests/ModelTests/modelstates/Final_Least_Squares_rotated_opt_model", "evaluation_objective" : "ncircles"}


## For now we'll hard code config and use our least squares model as a test

# Load Model
model = mo.ModelFactory.create(config["model_type"], {})
model.load_state_dict(torch.load(Path(io.find_project_root() / config["model_path"])))
model.eval()

# Load Data Sets
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = CircleData(
        csv_file=data_fp,
        target_columns=[
            "ncircles",
            "nbig"
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

with torch.no_grad():
    tr_analyzer = ma.AnalysisRunner(model, train_data, "./", config["evaluation_objective"])
    val_analyzer = ma.AnalysisRunner(model, val_data, "./", config["evaluation_objective"])
    tes_analyzer = ma.AnalysisRunner(model, test_data, "./", config["evaluation_objective"])

    print("#### TRAINING SET EVALUATION SETTINGS ###")
    tr_analyzer.obtain_schedule()

    print("####VALIDATION SET EVALUATION SETTINGS ###")
    val_analyzer.obtain_schedule()

    print("### TEST SET EVALUATION SETTINGS ###")
    tes_analyzer.obtain_schedule()

    print("Evaluating Training Data")
    tr_analyzer.run_schedule()
    print("Evaluating Validation Data")
    val_analyzer.run_schedule()
    print("Evaluating Test Data")
    tes_analyzer.run_schedule()

    print("Analysis Complete. Shutting Down.")
    tr_analyzer.close()
    val_analyzer.close()
    tes_analyzer.close()
