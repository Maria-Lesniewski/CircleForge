![A Circle Forge Feature Tensor. (size=600, max_circle = 20, fill_percent = 0.01)](doc/doc_images/sample_image.png)

# Circle Forge 

An end-to-end procedural image playground for experimenting with ML architectures under systematically varying data and modeling conditions. 
Its currently in progress, but each release has some minimum introduced functionality.  
The code and experiments are shared primarily as a technical portfolio and learning resource.

## What/Why is this?
CircleForge is a sandbox for experimenting with ML pipelines in a beginner-modeler + seasoned-scientist approachable way.
It is a totally controllable image classification pipeline, wherein synthetic circleforge data distributions can be infinitely sampled with controllable hidden parameters, while model architectures can be plug-n-play for each modeling task. 
The idea is to distill ML experience into concrete "what happens when" questions followed by generated answers. 

## V1
V1 of circle forge is hard-geared for image classification and regression questions.
It enables experiments where we can ask different questions about how modeling choices affect our reliability under different data conditions.
(e.g. different loss, training routines, modeling conditions) can be coordinated, altered one at a time, and logged to create a truly controlled scientific experiment, and sandbox for learning ML. 
It does so via modular class architectures that 

The experiments module currently contains a few tutorial questions and an intro to modeling together with blog post XXXX. 

The scripts module houses a complete ML pipeline, currently implemented by manually calling each script with experiment hyper parameters desired. 

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

### V1 Workflows
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

### DIY Workflows
Each driver script above interfaces with a particular class which is defined in the source folder. Usually classes are called in PascalCase and defined in a snake\_case.py src. 
Each src folder contains a ReadMe.md for interfacing with each class. 
