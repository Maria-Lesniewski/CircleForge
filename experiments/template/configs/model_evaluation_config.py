# This is a file for holding the custom model_evaluation configuration
# (for configuring scripts/analyze_models.py)
# paths are relative to the artifact directories 

CONFIG = {
        "model" : {
            "type" : "BasicLinear", # ModelFactoryKey string or "Ask" for prompt
            "meta_path" : "/home/maria/Documents/Coding_Space/Image_Generator_Project/artifacts/template_example_experiment/train_stats/BasicLinear_best_checkpoint_epochs_training/cpt_meta_data/BasicLinear_step_99.json", # meta_data written by Trainer class. or "Ask" for prompt
            },
        "data" : {
            "annotations_path" : "template_example_experiment/final_data/" + "train_val_test_split_annotations.csv",
            "target_columns" : ["ncircles", "nbig", "nsmall"],
            "feature_transform" : None
            },
        "analysis_settings" : {
            "evaluation_objective" : "ncircles", # str, colname
            "log_path" : "ModelEvaluation", # Results stored under modename in outside folder
            } # Currently the schedule must be piped in via the CL interface
        }
