#!/usr/bin/env bash

source ../../.venv/bin/activate

python run_pipeline1.py
python run_pipeline2.py
python run_pipeline2.py "$(realpath ../../artifacts/template_example_experiment/train_stats/BasicLinear_best_checkpoint_epochs_training/cpt_meta_data/BasicLinear_step_99.json)"
python run_pipeline3.py

bash grab_stats_matrices.sh 
