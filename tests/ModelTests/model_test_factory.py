# import pytest 
# import json Later it might make sense to go to json files if we add too many models
import torch
import torch.nn as nn
import Models.model_options as mo
import utils.io as io
import json
from pathlib import Path
class ModelTestRunner:

    """
    Checks forward function for a variety of model structures
    _modelstates -> registry for a fixed set of modelparameters used for testing
    _expected_output -> Numbers or vectors the model is supposed to create on a forward pass
    _configs -> args for each model type 
    """
    _checkpoints = {
            "BasicLinear" : "ModelTests/modelstates/BasicLinear.pt",
            "RotatedLinear" : "ModelTests/modelstates/RotatedLinear.pt",
            "Constant" : "ModelTests/modelstates/Constant.pt"
            }

    _expected_output = {
            "BasicLinear" : 37.6138, ### THIS IS THE BIAS PARAM,
            "RotatedLinear" : 38.3361, ### THIS IS THE BIAS PARM,
            "Constant" : 40.986219552886205, ### THIS IS THE CONSTANT PARAM
            }

    # For model classes with instantiating args
    # Current 3 are hard coded as none, but ideally we pass a dict with paramkeys and values
    _configs = {
            "BasicLinear" : {},
            "RotatedLinear" : {},
            "Constant" : {}
            }

    # Check the inits via model architecture
    # Architectures are stored in the test folder as json files
    _architectures = {
            "BasicLinear" : "ModelTests/modelarchitectures/BasicLinear.json",
            "RotatedLinear" : "ModelTests/modelarchitectures/RotatedLinear.json",
            "Constant" : "ModelTests/modelarchitectures/Constant.json",
            }

    _LAYER_TYPES = { # Used in .json architecture files
            "Flatten" : nn.Flatten,
            "Linear" : nn.Linear,
            "BLayer" : nn.Parameter,
            }

    @classmethod
    def test_forward(cls, model_type : str, input_data, tolerance=1e-12):
        model = mo.ModelFactory.create(
                model_type,
                cls._configs[model_type]
                )
        state = torch.load(cls._checkpoints[model_type])
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            actual = model(input_data)

        expected = cls._expected_output[model_type]
    
    # Use allclose so that maybe we can extend these tests for image generators later
        return torch.allclose(
                actual,
                torch.tensor(expected),
                atol=tolerance
                )
    @classmethod
    def test_model_init(cls, model_type : str):
        model = mo.ModelFactory.create(
                model_type,
                cls._configs[model_type]
                )
        # Check that layers are properly initialized
        layer_dict = cls.load_model_architectures(model_type)
        for expected_layer in layer_dict["layers"]:
            name = expected_layer["name"]
            expected_type = cls._LAYER_TYPES[expected_layer["type"]]
            
            assert hasattr(model, name), (
                    f"Model {model_type} is missing layer '{name}'"
                    )
            actual_layer = getattr(model, name)

            assert isinstance(actual_layer, expected_type), (
                    f"In Model {model_type}"
                    f"Layer '{name}' has type "
                    f"{type(actual_layer).__name__}, "
                    f"expected {expected_type.__name__}"
                    )

            expected_attributes = expected_layer.get("attributes")

            if expected_attributes:
                for attr, expected_value in expected_attributes.items():

                    if attr == "bias":
                        # Then we'll set to true if found
                        actual_value = (actual_layer.bias is not None)
                    else:
                        actual_value = getattr(actual_layer, attr)

                    assert actual_value == expected_value, (
                        f"Layer '{name}' attribute '{attr}' "
                        f"expected {expected_value}, "
                        f"got {actual_value}"
                    )
            return True

    @classmethod
    def load_model_architectures(cls, model_type : str):
        with Path(cls._architectures[model_type]).open() as f:
            return json.load(f)


