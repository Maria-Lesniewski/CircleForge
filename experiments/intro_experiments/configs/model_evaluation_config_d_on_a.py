# This is a file for holding the custom model_evaluation configuration
# (for configuring scripts/analyze_models.py)
# paths are relative to the artifact directories 

CONFIG = {
        "model" : {
            "type" : "BasicLinear", # ModelFactoryKey string or "Ask" for prompt
            "meta_path" : "intro_experiment/set_d/LeastSquaresResults/cpt_meta_data/BasicLinear_step_0.json", # meta_data written by Trainer class. or "Ask" for prompt (generally put your last latest which will vary dependent on data gen process)
            },
        "data" : {
            "annotations_path" : "intro_experiment/set_a/final_data/" + "train_val_test_split_annotations.csv",
            "target_columns" : ["ncircles"],
            "feature_transform" : None
            },
        "analysis_settings" : {
            "evaluation_objective" : "ncircles", # str, colname
            "log_path" : "intro_experiment/set_d/ModelEvaluation/set_a", # Results stored under modename in outside folder
            } # Currently the schedule must be piped in via the CL interface
        }
