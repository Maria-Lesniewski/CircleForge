# CircleForge API Documentation

This directory contains API documentation for the classes that make up the CircleForge ML pipeline.

Each page describes a class's purpose, public methods, attributes, and intended usage.

## API Reference

### Data Generation

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

- Model Performance is accomplished via the AnalysisRunner Class documented in the [ModelEvaluation Suite](model_evaluation.md). The classes within analyze characteristics of a particular model's predictions

- Properties of each DataSet can be measured using the DiagnosticsRunner Class documented in the [DataSetEvaluation Suite](data_set_evaluation.md). The classes within analyze characteristics of a particular dataset to help better design features and architectures within models.  
