# This is a file for holding the default model_evaluation configuration
# (for configuring scripts/analyze_models.py)
# paths are relative to the artifact directories 

DEFAULT_CONFIG = {
        "model" : {
            "type" : "Ask",
            "meta_path" : "Ask",
            },
        "data" : {
            "annotations_path" : "final_data/" + "train_val_test_split_annotations.csv",
            "target_columns" : ["ncircles", "nbig", "nsmall"],
            "feature_transform" : None
            },
        "analysis_settings" : {
            "evaluation_objective" : "ncircles", # str, colname
            "log_path" : "ModelEvaluation", # Results stored under modename in outside folder
            } # Currently the schedule must be piped in via the CL interface
        }
