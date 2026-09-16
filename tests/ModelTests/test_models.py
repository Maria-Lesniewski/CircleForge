import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import torch
from model_test_factory import ModelTestRunner
from Modeling.model_options import ModelFactory
import pytest


# This lets us create a list of simultaneous params to run the tests throgh 
@pytest.mark.parametrize(
        "model_type",
        [
            "BasicLinear",
            "RotatedLinear",
            ],
        )

def test_model_forward(model_type):
    input_data= torch.zeros(1,50,50)
    result = ModelTestRunner.test(
            model_type,
            input_data
            )

    if not result:
        print(f"\nFAILED MODEL: {model_type}")

    assert result
