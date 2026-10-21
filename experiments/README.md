# How to Experiment
See experiments/template for example configuration and runner files.
CircleForge supports out the box experimenting by specifying a configuration
for any of the steps of an experimental pipeline. Broken up into pieces wherein
its best to take a step back and sanity check the logged results
Though the syntax in each python "runner" file can be borrowed for customizing any piece of the pipeline

# Python Experiment Runner Scripts
--- Pipeline Part 1 ---
1) Sample Generation - calls scripts/generate_data with generator_config.py passed
2) DataSet Curation - calls scripts/scrub_and_split_data with split_config.py passed
3) DataSet Analysis- calls scripts/analyze_data with data_config.py passed

--- Pipeline Part 2 ---
4) ModelTraining (and re-training if desired) - calls scripts/train_models with training_config.py passed
(Can optionally launch a part 2 with altered configuration from a baseline model by calling pipeline2 with the model's metadirectory path)

--- Pipeline Part3 --
5) Model Evaluation - calls scripts/evaluate_models with evaluation_config.py passed

