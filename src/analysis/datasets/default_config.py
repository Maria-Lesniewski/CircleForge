# This is a file for holding the default data analysis configuration
# (for configurating scripts/analyze_data_splits.py)
# paths are relative to the artifact directories 

DEFAULT_CONFIG = {
        "annotations_path" : "final_data/train_val_test_split_annotations.csv",
        "target_columns" : ["ncircles", "nbig", "nsmall"],
        "analysis_write_path" : "DataDiagnostics",
        "feature_transform" : None
        } # Training schedule and configs are piped through input files
