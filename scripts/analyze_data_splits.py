from dataset.circle_data import CircleData
import torch
import torch.nn as nn
import analysis.datasets as de
import models.model_options as mo
import utils.io as io
from pathlib import Path
from analysis.datasets.default_config import DEFAULT_CONFIG

config=None
args = io.parse_config_args()
if (args.config is not None):
    config = io.load_config_from_CL(args.config)
else:
    config = DEFAULT_CONFIG

## UNPACK CONFIG
annotations_path = io.artifact_path(config["annotations_path"])
cols = config["target_columns"]
rel_write_path = config["analysis_write_path"]
tfunc = config["feature_transform"]
# Load Data Sets
train_data = CircleData(
        csv_file=annotations_path,
        target_columns=cols,
        split_tag="train",
        transform=tfunc,
        )
val_data = CircleData(
        csv_file=annotations_path,
        target_columns=cols,
        split_tag="val",
        transform=tfunc,
        )
test_data = CircleData(
        csv_file=annotations_path,
        target_columns=cols,
        split_tag="test",
        transform=tfunc,
        )

with torch.no_grad():
    io.make_artifact_dir(rel_write_path)
    analysis_path = io.artifact_path(rel_write_path)
    tr_analyzer = de.DiagnosticsRunner(train_data, analysis_path)
    val_analyzer = de.DiagnosticsRunner(val_data, analysis_path)
    tes_analyzer = de.DiagnosticsRunner(test_data, analysis_path)

    print("#### TRAINING SET Diag SETTINGS ###")
    tr_analyzer.obtain_schedule()
    print("####VALIDATION SET Diag SETTINGS ###")
    val_analyzer.obtain_schedule()

    print("### TEST SET Diag SETTINGS ###")
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
