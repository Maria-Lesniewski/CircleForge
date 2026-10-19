# CircleForge API Documentation

Each page within this directory describes a class's purpose, public methods, attributes, and intended usage.

## Overview of Pre-configured Pipelines 
CircleForge provides building blocks for a complete modeling problem pipeline. Classes coordinate PyTorch structures into recordable experimental pieces. 

A typical "experiment" involves creating a dataset, feature engineering from dataset statistics, creating a model structure, training the model, and evaluating its final performance.

Experiment results are typically stored in a generated `artifacts` folder at the project root. This usually includes direct results (e.g. model.pt files, analysis graphs, datasets) and also metadata (e.g. the conditions for which a dataset was generated.)

### <u>Data Generation and Storage</u>
Two classes are responsible for respectively procedurally generating image data and storing it for PyTorch training.

1) [`CircleDoodle` (CircleForge.generator)](circle_doodle.md) - Generates procedural circle patterns and tracks image annotations, including circle counts and area statistics.

2) [`CircleData` (CircleForge.dataset)](circle_data.md) - Dataset iterable, defined by image annotations and the linked .npy image vector files. Used with a dataloader during training and evaluation

Cleaning and splitting generated data is currently implemented more manually, see e.g. `scripts/scrub_split_data.py` for an example.  

### <u>Calling Models</u> 

Experiments with different models can be centralized through a [`ModelFactory`] interface, new models are registered with the factory and implemented as torch.nn.modules individually.

- [`ModelFactory`](model_factory.md) - A centralized registry for creating instances of PyTorch models via the same interface, used to write general training/evaluation code without dependency on model architecture

- The [Model Classes](models.md) page contains the list and documentation for currently implemented in CircleForge.models.model_options

### <u>Training Models</u>

Training hyperparameters (epoch routines, optimizers, loss functions) and model checkpointing are coordinated and logged through the [`Trainer`](trainer.md) class.

### <u>Statistical Analysis</u>

Analysis is usually conducted before and after a model is built during an experimentation pipeline. 
CircleForge.analysis houses multiple classes for analyzing model performance and dataset trends are available in the CircleForge.analysis module. 
In each case, statistics are aggregated simultaneously from streamed data and finalized to .xy grace files and tensor board scalar plots. 
For convenience, we also provide a `utils/grace_like.py` script which can be called on the command line for plotting .xy files passed with matplot lib. See the comments at the top of the `utils/grace_like.py` file for usage details.

The CircleForge.analysis module has two halves:

- CircleForge.analysis.models : 

    Model Evaluation in CircleForge is coordinated through [`AnalysisRunner`](runner_classes.md#AnalysiRunner).

    - The [`AnalysisRunner`](runner_classes.md#analysisrunner) class stores pre-registered analysis functions in [`AnalysisRegistry`](analysis_registry.md). The functions are calculated batch-wise while an iterator passes through a given dataset, allowing one model evaluation per iteration, regardless of the number of statistics calculated. See the 

- CircleForge.analysis.datasets:

    Data analysis in CircleForge is coordinated through  [`DiagnosticsRunner`](runner_classes.md#diagnosticsrunner). 

    - The [`DiagnosticsRunner`](runner_classes.md#diagnosticsrunner.md) stores pre-registered analysis functions in [`DiagnosticRegistry`](diagnostics_registry.md). These functions are calculated batch-wise while an iterater passes through a given dataset. See the [`DiagnosticsRunner`](runner_classes.md#diagnosticsrunner.md) class doc for usage details. See the [`DiagnosticsRegistry`](diagnostics_registry.md) class doc for the list of all implemented functions.


## Other Utilities for DIY experimenting
- [`CircleForge.utils.io`](io.md) provides an interface for anchoring `artifact` paths and prompting CL responses to configuration questions

- `CircleForge.utils.model_logger` stores the [`Logger`](logger.md) class, which is used in the coordinated structures above, but is also handy for saving custom data. 
