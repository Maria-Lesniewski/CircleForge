# CircleForge API Documentation

This directory contains API documentation for the classes that make up the CircleForge ML pipeline.

Each page describes a class's purpose, public methods, attributes, and intended usage.

## API Reference

### Data Generation

Classes responsible for procedurally generating synthetic images and their associated labels.

- [CircleDoodle](circle_doodle.md) - Generates procedural circle patterns and tracks image annotations, including circle counts and area statistics.

- [CircleData](circle_data.md) - Dataset iterable, defined by image annotations and the linked .npy image vector files. Used with a dataloader during training and evaluation

### Modeling 

Classes for Statistical Inference:

- [ModelFactory](model_factory.md) - A centralized registry for creating instances of PyTorch models via the same interface, used to write general training/evaluation code without dependency on model architecture

- [ModelClasses(torch.nn.module)](models.md) - Models that implement the torch.nn.module with different architecture, used for statistical comparison and inference 




