# Logger API Documentation

## Overview

The `Logger` class provides a unified interface for recording experimental metrics, saving model checkpoints, and storing metadata during machine learning experiments.

It supports the following functionalities:
- **Metric logging:** Records scalar values in `.xvg` files for external analysis and in TensorBoard for visualization.
- **Model checkpointing:** Saves model state dictionaries at specified training steps.
- **Checkpoint metadata:** Optionally stores model statistics and checkpoint paths in JSON files.
- **JSON serialization:** Saves arbitrary Python dictionaries as JSON files within the experiment directory.

The class is designed to simplify experiment logging and allow training scripts to manage experiment-relative paths directly.

## Directory Structure

Given a root directory `root_path`, the logger organizes its output as follows:

```text
root_path/
├── tensorboard/
│   └── events.out.tfevents.*
├── grace/
│   ├── <tag>_<name>.xvg
│   └── ...
├── checkpoints/
│   ├── <model_id>_step_<step>.pt
│   └── ...
├── cpt_meta_data/
│   ├── <model_id>.json
│   └── ...
└── <relative_filepath>.json
```

- `tensorboard/`: Contains TensorBoard event files.
- `grace/`: Contains scalar metric files in `.xvg` format, suitable for plotting and external analysis.
- `checkpoints/`: Contains serialized model state dictionaries.
- `cpt_meta_data/`: Contains optional JSON metadata associated with model checkpoints.
- `<relative_filepath>.json`: Represents arbitrary JSON files saved through `write_json()`. These files can be located anywhere beneath `root_path`.

The `grace/` and `checkpoints/` directories are created as needed. The `cpt_meta_data/` directory is created only when checkpoint metadata is written.

**Note:** The `Logger` does not automatically create the root directory explicitly. TensorBoard's `SummaryWriter` handles its own output directory, while other directories are created when required.

---

## Class: `Logger`

```python
class Logger(root_path: Path)
```

Provides methods for recording experiment metrics, saving model checkpoints, and writing JSON data.

### Constructor

```python
Logger(root_path: Path)
```

Initializes a logger for an experiment.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `root_path` | `Path` | Root directory where experiment logs, checkpoints, and metadata are stored. |

**Attributes:**

| Name | Type | Description |
|---|---|---|
| `root_path` | `Path` | Root directory for all experiment outputs. |
| `torchwriter` | `SummaryWriter` | TensorBoard writer used to record scalar metrics. |
| `grace_path` | `Path` | Directory containing `.xvg` scalar log files. |
| `checkpoint_path` | `Path` | Directory containing model checkpoints. |
| `meta_path` | `Path` | Directory containing checkpoint metadata JSON files. |

**Example:**

```python
from pathlib import Path

logger = Logger(Path("experiments/experiment_001"))
```

This initializes a logger that stores experiment outputs under `experiments/experiment_001/`.

---

## Methods

### `add_scalars()`

```python
add_scalars(
    tag: str,
    scalars: dict[str, float],
    step
) -> None
```

Records scalar metrics in both `.xvg` files and TensorBoard.

Each scalar is appended to a separate `.xvg` file. The filename is constructed from the provided `tag` and scalar name.

Each row in an `.xvg` file contains two values:

```text
step scalar_value
```

The method also forwards the scalar dictionary to TensorBoard using `SummaryWriter.add_scalars()`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `tag` | `str` | Label identifying the group of scalar metrics, such as `training` or `validation`. |
| `scalars` | `dict[str, float]` | Dictionary mapping scalar names to their values. |
| `step` | Numeric | Training step, batch index, or epoch associated with the scalar values. |

**Returns:**

- `None`

**Output files:**

For a call with `tag="training"` and the scalar dictionary `{"loss": 0.25, "accuracy": 0.92}`, the logger appends values to:

```text
grace/training_loss.xvg
grace/training_accuracy.xvg
```

Each file contains the corresponding step and scalar value.

**Example:**

```python
logger.add_scalars(
    tag="training",
    scalars={
        "loss": 0.25,
        "accuracy": 0.92,
    },
    step=100,
)
```

The resulting `.xvg` files contain:

```text
100.0 0.25
```

and:

```text
100.0 0.92
```

The actual formatting is determined by NumPy's `savetxt()` function.

**Notes:**
- Existing `.xvg` files are appended to rather than overwritten.
- A separate file is created for each scalar name.
- Parent directories for the generated `.xvg` files are created automatically.
- Scalar metrics are also logged to TensorBoard under the specified tag.
- The `step` argument is not explicitly type-annotated in the implementation.

---

### `close()`

```python
close() -> None
```

Flushes and closes the TensorBoard writer and removes the `grace/` directory if it exists and is empty.

**Parameters:**

- None.

**Returns:**

- `None`

**Behavior:**

1. Flushes any pending TensorBoard events.
2. Closes the TensorBoard writer.
3. Removes the `grace/` directory if it is empty.

The `grace/` directory is removed only when it contains no files. TensorBoard output is retained.

**Example:**

```python
logger.close()
```

**Important:** Call `close()` when logging is complete to ensure TensorBoard events are flushed and resources are released. The method should be called even if no scalar metrics were recorded.

---

### `save_model()`

```python
save_model(
    model_id: str,
    step: int,
    model: torch.nn.Module,
    stats: dict[str, Any],
    metadata: bool
) -> None
```

Saves a model's state dictionary as a PyTorch checkpoint. Optionally, it also writes a JSON file containing model metadata and statistics.

The checkpoint filename is constructed using the model identifier and training step:

```text
checkpoints/<model_id>_step_<step>.pt
```

The method saves the model using `torch.save(model.state_dict(), path)`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model_id` | `str` | Identifier used to name the model checkpoint and metadata file. |
| `step` | `int` | Training step, batch index, or epoch associated with the checkpoint. |
| `model` | `torch.nn.Module` | PyTorch model whose state dictionary will be saved. |
| `stats` | `dict[str, Any]` | Dictionary containing statistics or other metadata associated with the checkpoint. Used only if `metadata=True`. |
| `metadata` | `bool` | If `True`, writes a JSON metadata file alongside the checkpoint directory structure. If `False`, saves only the checkpoint. |

**Returns:**

- `None`

**Output files:**

For `model_id="classifier"` and `step=1000`, the checkpoint is saved to:

```text
checkpoints/classifier_step_1000.pt
```

If `metadata=True`, an additional file is written to:

```text
cpt_meta_data/classifier.json
```

**Example:**

```python
logger.save_model(
    model_id="classifier",
    step=1000,
    model=model,
    stats={
        "loss": 0.12,
        "accuracy": 0.95,
    },
    metadata=True,
)
```

**Loading a checkpoint:**

Since only the model's state dictionary is saved, the model architecture must be instantiated separately before loading the checkpoint.

```python
checkpoint_path = (
    logger.checkpoint_path / "classifier_step_1000.pt"
)

model.load_state_dict(
    torch.load(checkpoint_path, weights_only=True)
)
```

The `weights_only=True` argument is supported by recent PyTorch versions. For compatibility with older versions, consult the documentation for the installed PyTorch release.

**Notes:**
- The checkpoint directory is created automatically if it does not exist.
- Only the model's state dictionary is saved; the optimizer state, scheduler state, and training configuration are not saved automatically.
- If `metadata=False`, no metadata file is written.
- Metadata files are named using `model_id` alone, not `step`. Saving metadata for the same `model_id` at a later step overwrites the previous metadata file.
- The `stats` argument is ignored when `metadata=False`.

---

### `_write_model_metadata()`

```python
_write_model_metadata(
    model_id: str,
    step: int,
    stats: dict[str, Any]
) -> None
```

Writes checkpoint metadata to a JSON file.

This is a private helper method intended to be called internally by `save_model()` when metadata recording is enabled.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `model_id` | `str` | Identifier of the model. |
| `step` | `int` | Training step associated with the checkpoint. |
| `stats` | `dict[str, Any]` | Dictionary containing model statistics and other metadata. |

**Returns:**

- `None`

**Output file:**

```text
cpt_meta_data/<model_id>.json
```

The generated JSON document contains the following fields:

| Field | Type | Description |
|---|---|---|
| `model_id` | `str` | Identifier of the model. |
| `stats` | `dict` | Statistics and metadata provided to `save_model()`. |
| `checkpoint_path` | `str` | Path to the corresponding checkpoint file. |

**Example metadata:**

```json
{
    "model_id": "classifier",
    "stats": {
        "loss": 0.12,
        "accuracy": 0.95
    },
    "checkpoint_path": "experiments/experiment_001/checkpoints/classifier_step_1000.pt"
}
```

The actual checkpoint path depends on the configured root directory.

**Notes:**
- The metadata directory is created automatically.
- The JSON file is overwritten if a metadata file for the same model identifier already exists.
- The method uses `json.dump()` without a custom serializer. Values in `stats` must therefore be JSON-serializable.

---

### `write_json()`

```python
write_json(
    rel_filepath: str,
    dictionary: dict[Any, Any]
) -> None
```

Serializes a Python dictionary to a JSON file located beneath the logger's root directory.

The specified relative filepath is joined with `root_path`. Any necessary parent directories are created automatically.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `rel_filepath` | `str` | Relative path of the JSON file within the experiment directory. |
| `dictionary` | `dict[Any, Any]` | Dictionary to serialize and save. |

**Returns:**

- `None`

**Example:**

```python
logger.write_json(
    "config/training_config.json",
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "Adam",
    },
)
```

The output file is created at:

```text
<root_path>/config/training_config.json
```

Its contents are formatted as follows:

```json
{
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "Adam"
}
```

**Notes:**
- Parent directories are created automatically.
- Existing files at the specified path are overwritten.
- JSON output is indented by four spaces.
- The `default=str` argument allows otherwise non-serializable values to be converted to strings.
- The method does not enforce that `rel_filepath` remains within `root_path`. Absolute paths or paths containing `..` may resolve outside the experiment directory.
- JSON object keys must satisfy the requirements of Python's JSON encoder.

---

## Complete Usage Example

The following example demonstrates how to initialize the logger, record training metrics, save a model checkpoint, write configuration data, and close the logger.

```python
from pathlib import Path
import torch

# Initialize model and logger
model = MyModel()

logger = Logger(
    root_path=Path("experiments/experiment_001")
)

# Record training metrics
for epoch in range(10):
    train_loss = 0.5 / (epoch + 1)
    validation_loss = 0.6 / (epoch + 1)

    logger.add_scalars(
        tag="loss",
        scalars={
            "train": train_loss,
            "validation": validation_loss,
        },
        step=epoch,
    )

    # Save model checkpoint and metadata
    logger.save_model(
        model_id="my_model",
        step=epoch,
        model=model,
        stats={
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "epoch": epoch,
        },
        metadata=True,
    )

# Save experiment configuration
logger.write_json(
    "config.json",
    {
        "epochs": 10,
        "model": "my_model",
    },
)

# Flush and close the logger
logger.close()
```

Replace `MyModel` with the actual model class and the example metric calculations with the relevant training and validation logic.

---

## Dependencies

The implementation relies on the following Python packages and standard-library modules:

| Dependency | Purpose |
|---|---|
| `torch` | Model checkpoint serialization and PyTorch model types. |
| `torchvision` | Imported by the module but not used by the `Logger` implementation shown. |
| `numpy` | Writing scalar values to `.xvg` files. |
| `tensorboard` | Required by PyTorch's `SummaryWriter` for TensorBoard logging. |
| `pathlib` | Cross-platform path handling. |
| `typing` | Type annotations. |
| `json` | Writing metadata and configuration files. |
| `os` | Checking and removing empty directories. |

## Implementation Notes and Limitations

- **Resource management:** Always call `close()` after logging is complete. The class does not implement context-manager methods such as `__enter__()` and `__exit__()`.
- **Checkpoint contents:** Checkpoints contain only model parameters and buffers from `model.state_dict()`. They do not include optimizer state, scheduler state, random-number generator state, or other training information unless these are explicitly saved separately.
- **Metadata overwriting:** Each model identifier maps to a single metadata JSON file. Saving multiple checkpoints with the same `model_id` and `metadata=True` overwrites the metadata from earlier calls, even though the checkpoint files remain separate.
- **JSON serialization:** `_write_model_metadata()` does not use `default=str`, unlike `write_json()`. Non-JSON-serializable values in `stats` may therefore raise a serialization error.
- **Metric file format:** `.xvg` files are append-only and contain two columns: step and scalar value. They do not include headers or descriptive metadata.
- **TensorBoard logging:** TensorBoard logs are stored under `root_path / "tensorboard"`, independently of the `.xvg` files.
- **Directory cleanup:** `close()` removes only an empty `grace/` directory. It does not delete checkpoints, metadata, or TensorBoard logs.
- **Path handling:** Paths are constructed relative to `root_path`, but the class does not validate that user-provided relative paths remain within that directory.
- **Unused import:** `torchvision` is imported but is not used in the shown implementation and can be removed if it is not required elsewhere in the module.

# See Also

- [The io suite](io.md) - This can be paired with the logger for easy filepath handling / logging