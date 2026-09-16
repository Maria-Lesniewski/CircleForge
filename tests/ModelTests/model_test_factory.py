# import pytest 
# import json Later it might make sense to go to json files if we add too many models
import torch
import Modeling.model_options as mo

class ModelTestRunner:

    """
    Checks forward function for a variety of model structures
    _modelstates -> registry for a fixed set of modelparameters used for testing
    _expected_output -> Numbers or vectors the model is supposed to create on a forward pass
    _configs -> args for each model type 
    """
    _checkpoints = {
            "BasicLinear" : "ModelTests/modelstates/BasicLinear.pt",
            "RotatedLinear" : "ModelTests/modelstates/RotatedLinear.pt"
            }

    _expected_output = {
            "BasicLinear" : 37.6138, ### THIS IS THE BIAS PARAM,
            "RotatedLinear" : 38.3361 ### THIS IS THE BIAS PARM
            }

    # For model classes with instantiating args
    # Current 2 are hard coded as none, but ideally we pass a dict with paramkeys and values
    _configs = {
            "BasicLinear" : {},
            "RotatedLinear" : {}
            }

    @classmethod
    def test(cls, model_type : str, input_data, tolerance=1e-12):
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

