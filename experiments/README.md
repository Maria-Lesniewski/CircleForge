**How to Experiment**

- `experiments/template` contains example configuration and runner files that can be run end-to-end to create a simple model and analyze its performance. 
- Configuration files can be altered directly and contain comments describing intended format
- Generally, an end to end pipeline is created piecewise by gaining information from any of three parts and using that to inform the next step or to sanity check results before dedicating more resources. We've created three natural pieces in the example runner files. 

- The syntax in each python "runner" file can be borrowed for customizing any piece of the pipeline in future experiments

**Runner Scripts**
 - Part 1: 

    1) Sample Generation - calls `scripts/generate_data.py` with `generator_config.py` passed
    2) DataSet Curation - calls `scripts/scrub_and_split_data.py` with `split_config.py` passed
    3) DataSet Analysis- calls `scripts/analyze_data_splits.py ` with `data_config.py` passed

- Part 2:

    4) ModelTraining (and re-training if desired) - calls `scripts/train_models.py` with training_config.py passed
    (Can optionally launch a part 2 with altered configuration from a baseline model by calling pipeline2 with the model's metadirectory path)

- Part 3:

    5) Model Evaluation - calls `scripts/evaluate_models.py` with `evaluation_config.py` passed

**Tutorial Requirements**

Each tutorial runs entirely on CPUS, though with slightly different hardware reqs:
    - Experiment 1: intro_experiment:
        - tiny data set, (<1 GB RAM req, 1 CPU)
    - Experiment 2: feature_engineering:
        - Larger data set, (up to 8 GB RAM employed across workflow, 4 CPUs for data generation)
        - (don't have 4 CPUs? Change NUM_PROCESSES in run_pipeline1 and the "data" num_workers parameters in training)
    - Experimetn 3: nn_training:
        - Same dataset as experiment 2. Up to 8 GB of RAMP employed during gradient descent., 4 workers on data loader. 