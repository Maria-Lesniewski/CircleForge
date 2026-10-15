import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import torch
from model_test_factory import ModelTestRunner
from models.model_options import ModelFactory
import pytest


# This lets us create a list of simultaneous params to run the tests throgh 

MODEL_TUPLE = ("model_type", 
        [
            "BasicLinear",
            "RotatedLinear",
            "Constant"
            ]
        )

@pytest.mark.parametrize(*MODEL_TUPLE)
def test_model_forward(model_type):
    input_data= torch.zeros(1,50,50)
    result = ModelTestRunner.test_forward(
            model_type,
            input_data
            )

    if not result:
        print(f"\nFAILED MODEL: {model_type}")

    assert result

@pytest.mark.parametrize(*MODEL_TUPLE)
def test_inits(model_type):
    result = ModelTestRunner.test_model_init(model_type)

    if not result:
            print(f"\n FAILED MODEL INITIALIZATION: {model_type}")
    
    assert result


