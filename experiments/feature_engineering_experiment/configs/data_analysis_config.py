# This is a file for holding custom data analysis configuration
# (for configurating scripts/analyze_data_splits.py)
# paths are relative to the artifact directories 

CONFIG = {
        "annotations_path" : "feature_engineering_experiment/final_data/" + "train_val_test_split_annotations.csv",
        "target_columns" : ["ncircles", "nbig", "nsmall"],
        "analysis_write_path" : "feature_engineering_experiment/DataDiagnostics",
        "feature_transform" : None
        } # Training schedule and configs are piped through input files
