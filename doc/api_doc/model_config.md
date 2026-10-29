# ModelConfig
## Overview 

The `ModelConfig` class bundles a model instance with metadata describing its type, source configuration, previous training statistics, and saved model parameters. 
Its primary use is to help continue training routines from saved data and to assist in contextualizing source models during model evaluations (e.g. in [AnalysisRunner](runner_classes.md)
It supports initializing new models, loading models from previously saved configurations, and integrating non-CircleForge trained models into the framework. 

The class uses `ModelFactory` to construct models registered with the framework and PyTorch to load model paramters from checkpoints. 

`ModelConfig` supports three main workflows:
    
    - **New model initialization**: Creates a model of the specified type with its default or factory-defined initialization. (via __init__ with appropriate args)
    
    - **Checkpoint loading** : Loads model metadata from a JSON configuration file and restores model parameters from the associated checkpoint, allowing a previously trained model to be reused. (via __init__ with appropriate args)
    
    - **External model integration**: Wraps an existing model instance or constructs a model using specified arguments, optionally loading parameters from a checkpoint (via ModelConfig.create_from_module).

The class also provides methods for extracting information about the model's origin, determining whether the model is newly initialized, and serializing its configuration into a dictionary. 

## Attributes
Attribute | Type | Description | 
|---|---|---|
|`model_type` | `str`  | Identifies the type of model being configured. For models created through `ModelFactory`, this corresponds to a registered model type. | 
| `model` | `torch.nn.Module` | The actual PyTorch model instance associated with the configuration. |
| `model_meta` | `dict` | Stores metadata describing the model's origin and previous training information. |
| `model_meta["src_config_file"]` | `str` | The path to the source configuration JSON file, or "New" when the model is initialized without a previous configuration. |
| `model_meta["src_previous_data"]` | `str` | `dict` or `None` | Stores the training statistics extracted from the source configuration's "stats" field. For newly initialized or externally integrated models, this is initialized to `None`| 
| `model_meta["src_previous_pt"]` | `str`, `Path`, or `None` | Stores the path to a checkpoint containing the model parameters when available. For models loaded from a configuration file, the path is obtained from "checkpoint_path". For models integratied through `create_from_module()`, it is stored as a `Path` if a checkpoint is provided. |

## Methods 

- `__init__(model_type: str = None, src: Path = None)`

    <details>
    <summary>Initializes a ModelConfig instance from either a model type or a previously saved configuration file. </summary>

    Exactly one of model_type and src must be provided. If both are provided, or neither is provided, the constructor raises a ValueError.

    When initializing a new model, the constructor:

    - Stores the specified model type.
    - Sets model_meta["src_config_file"] to "New".
    - Initializes model_meta["src_previous_data"] to None.
    - Creates the model using ModelFactory.create().

    When loading from an existing configuration file, the constructor:

    - Stores the source configuration path.
    - Reads the JSON configuration file.
    - Extracts the previous training statistics from "stats".
    - Extracts the checkpoint path from "checkpoint_path".
    - Retrieves the model type from "model_id".
    - Creates the model using ModelFactory.create().
    - Loads the saved parameters into the model using torch.load() and load_state_dict().


    The source configuration file must contain the expected metadata fields for checkpoint loading to succeed.

    **Parameters:**

    - model_type (str, optional): The type of model to create. Used when initializing a new model without a previous configuration.
    - src (Path, optional): The path to a previously saved JSON configuration file containing model metadata and a checkpoint path.

    **Returns:**
    
     None. Initializes the instance in place.
    
    **Example:**
    ```python
    from pathlib import Path

    # Initialize a new model configuration.
    config = ModelConfig(model_type="MyModel")

    print(config.model_type)
    print(config.model)
    print(config.model_meta)

    # Example metadata:
    # {
    #     "src_config_file": "New",
    #     "src_previous_data": None
    # }
    ```

    **Example — Loading a previously configured model:**

    ```python

    from pathlib import Path

    config = ModelConfig(src=Path("artifacts/model_config.json"))

    print(config.model_type)
    print(config.model)
    print(config.model_meta)

    The JSON file is expected to contain fields such as:
    json

    {
        "model_id": "MyModel",
        "stats": {
            "epoch": 10,
            "loss": 0.025
        },
        "checkpoint_path": "artifacts/model_step10.pt"
    }
    ```

    The example metadata fields and values are illustrative; the actual contents depend on the configuration file that was saved.
    </details>
- `extract_config_step_info() -> int`
  <details>
    <summary>
    Extracts the step number associated with the checkpoint from the stored metadata.</summary>


    If the model was initialized from a previous configuration, the method extracts the step number from the filename stored in model_meta["src_previous_pt"].

    The implementation assumes that the checkpoint filename ends with a step identifier in the format step`<number>`.pt, where the number follows the final underscore.

    If the model was initialized as new, the method returns 0.

    **Parameters:** None.

    **Returns:** int. The extracted step number, or 0 if the model is newly initialized.

    **Example:**
    ```python

    # For a newly initialized model:
    config = ModelConfig(model_type="MyModel")

    step = config.extract_config_step_info()
    print(step)

    # Output:
    # 0
    ```
    **Example — Previously trained model:**
    ```python

    # Assume the configuration references:
    # artifacts/model_step25.pt

    config = ModelConfig(src=Path("artifacts/model_config.json"))

    step = config.extract_config_step_info()

    # Output:
    # 25
    ```

    Note: The method relies on the checkpoint filename following the expected naming convention. If the filename does not contain a valid integer step identifier in the expected position, the method may raise a ValueError.
    </details>

- `is_new() -> bool`
    <details>
    <summary>Determines whether the model configuration represents a newly initialized model or a model loaded from a previous configuration.</summary>

    The method checks whether model_meta["src_config_file"] equals "New".

    **Parameters:** None.

    **Returns:** bool.

    - True: if the model was initialized using model_type without a source configuration.
    - False: if the model was initialized using a source configuration file.

    **Example:**
    ```python

    config = ModelConfig(model_type="MyModel")

    print(config.is_new())

    # Output:
    # True
    ```
    **Example—For a model loaded from a previous configuration:**

    ```python

    config = ModelConfig(src=Path("artifacts/model_config.json"))

    print(config.is_new())

    # Output:
    # False
    ```

    Note: Models created through create_from_module() do not initialize model_meta["src_config_file"]. Consequently, calling is_new() on an instance created through that method may raise a KeyError.
    </details>

- `to_dict() -> dict[str, Any]`
    <details>
    
    <summary>Serializes the instance's top-level attributes into a dictionary.</summary>

    The method uses `vars(self)` to collect the instance attributes and their current values. The returned dictionary includes attributes such as model_type, model, and model_meta.

    This method provides a simple way to inspect or serialize the configuration structure. However, it does not itself convert the PyTorch model into a JSON-compatible representation.

    **Parameters:** None.

    **Returns:** dict[str, Any]. A dictionary containing the instance's top-level attributes and their values.

    **Example:**
    ```python

    config = ModelConfig(model_type="MyModel")

    config_dict = config.to_dict()

    print(config_dict.keys())

    # Expected keys include:
    # "model_type"
    # "model"
    # "model_meta"
    ```
    Note: Although the method is described as a serialization helper, its returned dictionary may contain a PyTorch model object and other values that cannot be directly serialized to JSON without additional processing.
    </details>

- `create_from_module(cls, model_type: str, model_params_path: Path = None, model: nn.Module = None, build_model_args: dict[str, Any] = None)`
    <details>
    <summary> Creates a ModelConfig instance for a model that has already been trained or initialized outside the standard model configuration workflow. </summary>

    This method is intended for integrating externally created models into the framework, including models for which no previous configuration metadata exists.

    Unlike the standard constructor, this method does not require a source configuration JSON file.

    The method performs the following operations:

    - Creates a ModelConfig instance without calling the standard constructor.
    - Stores the specified model type.
    - Initializes model_meta with "src_previous_data" set to `None`.
    - Uses the supplied model instance if model is provided.
    - Otherwise, creates a model using ModelFactory.create(model_type, build_model_args).
    - If model_params_path is provided, loads the checkpoint parameters into the model and records the checkpoint path in model_meta["src_previous_pt"].
    - Returns the resulting configuration instance.

    At least one of model or build_model_args must be provided. If both are omitted, the method raises a ValueError.

    **Parameters:**

    - model_type (str): The type identifier associated with the model.
    - model_params_path (Path, optional): The path to a checkpoint containing model parameters. If provided, the parameters are loaded into the model.
    - model (torch.nn.Module, optional): An existing PyTorch model instance to wrap in the configuration.
    - build_model_args (dict[str, Any], optional): Arguments passed to ModelFactory.create() when constructing a new model instance.

    **Returns:** ModelConfig. A configuration instance containing the model and its associated metadata.

    **Example — Wrapping an existing model:**
    ```python

    import torch.nn as nn

    # Assume model is an existing PyTorch model.
    model = nn.Linear(10, 1)

    config = ModelConfig.create_from_module(
        model_type="ExternalModel",
        model=model
    )

    print(config.model_type)
    print(config.model)
    print(config.model_meta)
    ```
    
    **Example — Constructing a model through ModelFactory:**
    ```python

    config = ModelConfig.create_from_module(
        model_type="MyModel",
        build_model_args={
            # Supply the arguments expected by the registered model factory.
        }
    )
    ```

    **Example — Loading parameters from a checkpoint:**
    ```python

    from pathlib import Path

    config = ModelConfig.create_from_module(
        model_type="MyModel",
        build_model_args={},
        model_params_path=Path("artifacts/model_step25.pt")
    )
    ```

    **Important implementation notes:**

    - If both model and build_model_args are provided, the supplied model takes precedence, and build_model_args is ignored.
    - If model_params_path is omitted, the implementation assigns None to "src_previous_pt".
    - The method does not initialize "src_config_file", so methods such as is_new() and extract_config_step_info() may fail when called on an instance created through this workflow.
    - Although model_params_path is annotated as a Path, the implementation converts it to a Path internally, so a path-like value such as a string may also work.
    - The checkpoint is loaded using torch.load() without explicitly specifying a device mapping. Loading checkpoints saved on a different device may therefore require additional handling.

## Usage Examples
**Creating a new model**

```python

config = ModelConfig(model_type="MyModel")

model = config.model

# Check whether the model is newly initialized.
print(config.is_new())

# Output:
# True
```

**Loading a model from a previous configuration**

```python

from pathlib import Path

config = ModelConfig(src=Path("artifacts/model_config.json"))

model = config.model

# Inspect the original model type and training statistics.
print(config.model_type)
print(config.model_meta["src_previous_data"])

# Extract the checkpoint step.
print(config.extract_config_step_info())
```

**Integrating an externally trained model**

```python
import torch.nn as nn

external_model = nn.Linear(10, 1)

config = ModelConfig.create_from_module(
    model_type="ExternalModel",
    model=external_model
)

model = config.model
```

## Notes and Limitations


- *Model registration*: Models constructed through ModelFactory.create() must be supported by the factory and its registered model types.

- *Constructor arguments*: The standard constructor accepts either `model_type` or `src`, but not both. It also does not accept a pre-existing model instance directly; use create_from_module() for that workflow.

- *Checkpoint format*: The standard constructor expects a JSON configuration containing "model_id", "stats", and "checkpoint_path".

- *Checkpoint loading*: Both model-loading workflows use torch.load() followed by load_state_dict(). The checkpoint must contain parameters compatible with the selected model architecture.

- *Metadata consistency*: The metadata fields differ between the standard constructor and create_from_module(). In particular, the latter does not initialize "src_config_file".

- *Serialization*: to_dict() returns the instance attributes without converting the model or nested metadata into a JSON-compatible format.

- *Step extraction*: extract_config_step_info() assumes a particular checkpoint filename convention and does not inspect checkpoint contents to determine the training step.

- *External models*: create_from_module() can wrap models that are not created by ModelFactory, provided the caller supplies the existing model instance.