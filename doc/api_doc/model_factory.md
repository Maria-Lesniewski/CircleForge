# Model Factory API

## Overview
`ModelFactory` provides a centralized registry for constructing PyTorch models by name

It allows training and evaluation code to instantiate models without dependending directly on individual model classes

Models are registered using the `@ModelFactor.register()` decorator and instantiated using `ModelFactory.create()`

## ModelFactory Methods
- `ModelFactory.register(name : str)`
    <details>
    <summary> Adds model class to the register stored within this class</summary>

    | Parameter | Type | Description |
    |---|---|---|
    |name| `str`| Name used to identify the model in the registry|

    **Returns**

    A decorator that registers the decorated class and returns the class unchanged

    **Example**
    ```
    @ModelFactory.register("MyModel")
    class MyModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = nn.Linear(10, hidden_size)

    def forward(self, x):
        return self.layer(x)
    ```
    Once registered, the model can be instantiated through the factory.
    </details>

- `ModelFactory.create(model_type : str, config : dict[str, Any] -> nn.Module)`
    <details>
    <summary> Creates an instance of a registered model.
    </summary>

    | Parameter | Type | Description |
    |---|---|---|
    |model_type | `str` | Registered name of the model to instantiate (see ModelFactory.Register)|
    |config | `dict[str, Any]` | keyword arguments passed to the model constructor |

    **Returns**
    An instance of the requested model as a PyTorch nn.Module

    **Raises**
    - ValueError: If `model_type` is not registered. 

    **Example**
    ```
    model = ModelFactory.create(
        model_type="BasicLinear",
        config={}
    )
    ```
    </details>

## Currently Registered Models

| Registry_Key | Class | Description |
|---|---|---|
|"BasicLinear" | BasicLinear | Linear regression over flattened image pixels. |
|"RotatedLinear" | RotatedLinear | Linear regression with rotational averaging over four right-angle orientations|
| "Constant" | Constant | A trainable constant prediction independent of image content |

## Design Notes
- Model constructors are called with keyword arguments unpacked from config
- Each model must inherit from torch.nn.Module
- The registry is populated when the module containing the model definitions is imported 

## See Also
- [Model Definitions](models.md)