![A Circle Forge Feature Tensor. (size=600, max_circle = 20, fill_percent = 0.01)](doc/doc_images/sample_image.png)

# Circle Forge 

An end-to-end procedural image playground for experimenting with ML architectures under systematically varying data and modeling conditions. 
Its currently in progress, but each release has some minimum introduced functionality.  
The code and experiments are shared primarily as a technical portfolio and learning resource.

## What/Why is this?
CircleForge is a configurable learning pipeline for experimenting with model structures under different data/training conditions.
It features totally controllable synthetic image database generation, a modular model factory for adding custom torch models, and a similar register for adding routines. 
Each experiment logs itself and its statistical analysis in an artifacts folder, so that experimenting in CircleForge is as easy as changing configuration files. 

The idea is to distill ML experience into concrete "what happens when" questions followed by generated answers. 

## V1
V1 of circle forge is geared for image regression with different architectures.
It introduces the first version of "configurable/logged" analysis, which enables experiments where we can ask different questions about how modeling choices affect our reliability without changing the infrastructure of the code.
(e.g. different loss, training routines, modeling structures) can be coordinated, altered one at a time, and logged to create a truly controlled scientific experiment, and sandbox for learning ML. 

The experiments module currently contains a few tutorial questions and an intro to modeling, while I am planning on making a more formal containerized tutorial and intro in the future as I expand the capability to more questions. 

For now: 
The scripts module houses a complete ML pipeline that can be called in python in order or run as sub-modules as in the experiments folder. Scripts launched with no --config or --model arguments will read a default configuration imported from the src. 

## Quick Start

### Installation & Test
Create/use a python virtual environment via: 
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

cd tests/
pytest
```

### Run the Template Experiment
```
cd experiments/template
bash end_to_end.sh
```

### Understand the workflow each pipeline script offers
- See the README.md in artifacts
- See the README.md in experiments
- Run the sub-module pipeline manually, interact with menus on the CL to understand the scope of abilities
- Read the DEFAULT\_CONFIG config files imported in each driver script 

### V1 Driver Scripts
Generally, a regression problem involves. 

1) Obtaining Data with labels
```
mpirun -np <X> python scripts/generate_training_data.py
```

2) Splitting Data into Train, Validation, and Test. Assure no overlap / duplicate samples
```
python scripts/scrub_and_split_data.py
```

3) Diagnose underlying correlations or patterns in your data, check for differences between splits.
```
python scripts/analyze_data_splits.py
```
4) Create a model and train it
```
python scripts/train_models.py
```
5) Evaluate how it went
```
python scripts/analyze_models.py
```
Each script uses the CL to ask you for different conditions as far as training, modeling and analaysis. Results are generated to an artifacts folder with the path announced for each. 

The experiments folder houses run scripts and config files that can be used to customize the parameters of each step with the pre-coded options. that come up when the main folder scripts are called without args. 

### DIY Workflows / Overview of code structure for custom experiments
Each driver script above interfaces with a particular class which is defined in the source folder. Usually classes are called in PascalCase and defined in a snake\_case.py src. 
Each src folder contains a README.md for interfacing with each class. 
