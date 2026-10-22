from dataset.circle_data import CircleData
import torch
import torch.nn as nn
import analysis.models as ma
import models.model_options as mo
import utils.io as io
from pathlib import Path
from analysis.models.default_config import DEFAULT_CONFIG
from models.model_config import ModelConfig

config=None
args = io.parse_config_args()
if (args.config is not None):
    config = io.load_config_from_CL(args.config)
else:
    config = DEFAULT_CONFIG

model_config = None
if config["model"]["type"] == "Ask" and config["model"]["meta_path"] == "Ask":
    context = ("Model analysis requires a model to load into the analyzer\n" + 
            "Please input a path to the requested model's .pt file\n" + 
            "We will infer the model type as XXX where the format should be <path>/XXX_YYY_ZZZ.pt\n"
            )
    choice = io.dictionary_filling_question(context, ["model_checkpoint_file"])
    modeltype = choice["model_checkpoint_file"].split("/")[-1].split("_")[0]
    model_config = ModelConfig.create_from_module(model_type=modeltype, model_params_path = choice["model_checkpoint_file"], build_model_args={})

elif config["model"]["meta_path"] != "Ask":
    model_config = ModelConfig(src = io.artifact_path(config["model"]["meta_path"]))
else:
    raise ValueError("Incompatible configuration file. Must request info via `Ask` in both `model` fields or link a relative model meta data path")

## UNPACK CONFIG
# model settings
model = model_config
model.model.eval()
### Data settings
data_fp = io.artifact_path(config["data"]["annotations_path"])
cols = config["data"]["target_columns"]
tfunc = config["data"]["feature_transform"]

### Analysis Settings
log_path = io.artifact_path(config["analysis_settings"]["log_path"])
eval_obj = config["analysis_settings"]["evaluation_objective"]


# Load Data Sets
train_data = CircleData(
        csv_file=data_fp,
        target_columns=cols,
        split_tag="train",
        transform=tfunc
        )
val_data = CircleData(
        csv_file=data_fp,
        target_columns=cols,
        split_tag="val",
        transform=tfunc,
        )
test_data = CircleData(
        csv_file=data_fp,
        target_columns=cols,
        split_tag="test",
        transform=tfunc,
        )

with torch.no_grad():
    tr_analyzer = ma.AnalysisRunner(model, train_data, log_path, eval_obj)
    val_analyzer = ma.AnalysisRunner(model, val_data, log_path, eval_obj)
    tes_analyzer = ma.AnalysisRunner(model, test_data, log_path, eval_obj)

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
