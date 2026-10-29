![A Circle Forge Feature Tensor. (size=600, max_circle = 20, fill_percent = 0.01)](doc/doc_images/sample_image.png) 

(A single CircleForge Data Point)

# CircleForge 

An end-to-end procedural image playground for experimenting with ML architectures under systematically varying data and modeling conditions. 
CircleForge is under active development, but each release has some minimum introduced functionality.  
The code and experiments are shared primarily as a technical portfolio and learning resource.

## What/Why is this?
CircleForge is a configurable learning pipeline for experimenting with model structures under different data/training conditions.
It features totally controllable synthetic image database generation, a modular model factory for adding custom torch models, and a similar register for adding training and analysis routines. 
ML experiments are configured with python dictionary files that control the parameters of pre-implemented routines for each part of the procedure. 
CircleForge will create logged data of each procedure in the artifacts folder for repeatable experiments and easy analysis.

The idea is to distill ML experience into concrete "what happens when" questions followed by generated answers, where the difference between experiments can boil down to a single parameter in a configuration file.  

## V1
The V1 release establishes the basic workflow for image regression with different architectures.
It introduces the first version of "configurable/logged" analysis, which enables experiments probing how different conditions impact the regression problem without changing the infrastructure of the code. 
(e.g. configurable files allow users to explore model architectures, objective functions, gradient descent parameters, data conditions)

**Current Workflow**

- The scripts module houses a complete ML pipeline that can be called in python in order or run as sub-modules as in the experiments folder. 
- Scripts launched with no --config or --model arguments will read a default configuration imported from the src. 
- The experiments folder houses three tutorial experiments along with template files for building your own, though I anticipate containerizing and writing a more formal tutorial in future releases.

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

## Understand the workflow each pipeline script offers
- See the [artifacts folder](artifacts/README.md) for an overview of experiment folder structures
- See the [experiments folder](experiments/README.md) for 3 introductory experiment examples 
- Run the sub-module pipeline in `scripts/` manually, interact with menus on the CL to understand the scope of abilities
- Read the DEFAULT\_CONFIG config files imported in each driver script 

### V1 Driver Scripts
Generally, a regression problem involves the following steps. 

1) Obtain data with labels
    ```
    mpirun -np <X> python scripts/generate_training_data.py
    ```

2) Split data into disjoint "training", "validation", and "test" subsets, ensuring no overlap or duplicate samples
    ```
    python scripts/scrub_and_split_data.py
    ```

3) Diagnose correlations or patterns in the data, check across split distributions.

    ```
    python scripts/analyze_data_splits.py
    ```
4) Create a model and train it

    ```
    python scripts/train_models.py
    ```

5) Evaluate the model on each split

    ```
    python scripts/analyze_models.py
    ```
Each script prompts the user on the CL to select any of the pre-implemented modeling/training/evaluation. 
When in doubt for generating a configuration file or running a sub-module, try calling a driver script and interacting with it to determine what options are already implemented. 

Each script will write to the artifacts folder in a configured or default path. 

## Custom (Non-Configurable) experiments

CircleForge implements a subset of all the possible ways to build and analyze different regression models,so it is designed with a modular architecture that make adding more models or analyses straight forward.  
Although for V1 I do not have a dev doc, the doc/api_doc can provide insight into this implementation which is housed in the src directory.

Each driver script above interfaces with a particular class which is defined in the source folder. Usually classes are called in PascalCase and defined in a snake\_case.py src, which is documented in a snake\_case.md file. 