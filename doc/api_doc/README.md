# CircleForge API Documentation

This directory contains API documentation for the classes that make up the CircleForge ML pipeline.

Each page describes a class's purpose, public methods, attributes, and intended usage.

## API Reference

### Data Generation and Storage

Classes responsible for procedurally generating synthetic images and their associated labels.

- [CircleDoodle](circle_doodle.md) - Generates procedural circle patterns and tracks image annotations, including circle counts and area statistics. 

- [CircleData](circle_data.md) - Dataset iterable, defined by image annotations and the linked .npy image vector files. Used with a dataloader during training and evaluation

### Modeling 

**Model Architectures**

Classes for Statistical Inference:

- [ModelFactory](model_factory.md) - A centralized registry for creating instances of PyTorch models via the same interface, used to write general training/evaluation code without dependency on model architecture

- [ModelClasses(torch.nn.module)](models.md) - Models that implement the torch.nn.module with different architecture, used for statistical comparison and inference 

**Training Models**

- [Trainer](trainer.md) - The Trainer class coordinates the particular training parameters for a given model, e.g. the loss function, learning rate, optimizer, epoch style, and logging (model checkpointing, training hyper parameters, loss value etc.) of a given model's training routine.   

### Statistical Analysis

Multiple classes for statistical analysis of model performance and dataset diagnostics are available in the CircleForge.analysis module. Statistics are aggregated simultaneously from streamed data and finalized to .xy grace files and tensor board scalar plots. 

- The [AnalysisRunner](model_evaluation.md) Class coordinates statistical analysis for evaluating model performance. Analysis are defined as classes within the `analysis.models`  package.

- The [DiagnosticsRunner](data_diagnostics.md) Class coordinates statistical analysis for diagnosing features/trends present in a particular dataset. Diagnostics are defined as classes within the `analysis.datasets` package.