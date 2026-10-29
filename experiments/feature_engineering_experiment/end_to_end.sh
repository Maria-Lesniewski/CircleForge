#!/usr/bin/env bash

source ../../.venv/bin/activate

python run_pipeline1.py
python run_pipeline2.py
python run_pipeline3.py

bash grab_stats_matrices.sh 
