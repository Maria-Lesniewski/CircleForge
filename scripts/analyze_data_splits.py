from DataSet.circle_data import CircleData
import torch
import torch.nn as nn
import StatisticalAnalysis.DataSetEvaluation as de
import Models.model_options as mo
import utils.io as io
from pathlib import Path

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
        target_columns=["ncircles",
            ],
        split_tag="val"
        )
test_data = CircleData(
        csv_file=data_fp,
        target_columns=["ncircles",
            "nbig",
            "nsmall"
            ],
        split_tag="test"
        )

with torch.no_grad():
    tr_analyzer = de.DiagnosticsRunner(train_data, "./")
    val_analyzer = de.DiagnosticsRunner(val_data, "./")
    tes_analyzer = de.DiagnosticsRunner(test_data, "./")

    print("#### TRAINING SET Diag SETTINGS ###")
    tr_analyzer.obtain_schedule()
    print("####VALIDATION SET Diag SETTINGS ###")
    val_analyzer.obtain_schedule()

    #print("### TEST SET Diag SETTINGS ###")
    #tes_analyzer.obtain_schedule()
    
    print("Evaluating Training Data")
    tr_analyzer.run_schedule()
    print("Evaluating Validation Data")
    val_analyzer.run_schedule()
    
    #print("Evaluating Test Data")
    #tes_analyzer.run_schedule()
    
    print("Analysis Complete. Shutting Down.")
    tr_analyzer.close()
    val_analyzer.close()
    # tes_analyzer.close()
