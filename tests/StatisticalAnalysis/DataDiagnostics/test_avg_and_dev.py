import pytest 
import utils.io as io
from test_utils.toy_database import ToyDatabase
from test_utils.spy_logger import SpyLogger
from StatisticalAnalysis.DataSetEvaluation.diagnostics_library.avg_and_dev import avg_and_dev
from StatisticalAnalysis.streaming_state import StreamingState
import pandas as pd
import torch
import json

### SET UP ####
### Fixtures and Setup for accessing standards ###

KEYS = [
        "image_size",
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units",
        "fill_fraction",
        ]

@pytest.fixture(scope="module")
def run_standards():
    with open(io.find_here() / "fixtures" / "avg_and_dev.running.json", "r") as f:
        data = json.load(f)

        # I didnt store N since its the same for every label 
        for ele in KEYS:
            data[ele]["N"] = 240

    return data

@pytest.fixture(scope="module")
def final_standards():
    data = {}
    for ele in KEYS:
        with open(io.find_here() / "fixtures" / f"avg_and_dev.final.{ele}.json", "r") as f:
            data[ele] = json.load(f)
    return data

@pytest.fixture(scope="module")
def toy_database():
    """
    Generate the toy database once for all tests and remove afterword
    """
    db = ToyDatabase("./")
    db.generate()

    yield db # Tests run to here

    db.remove() # Post test cleanup

@pytest.fixture(scope="module")
def streamed_data_obj(toy_database):
    """
    Loads toy database into streaming object 
    to interface with analysis functions
    """
    df = pd.read_csv("toy_annotations.csv")
    y = {}
    for ele in KEYS:
        y[ele] = torch.tensor(df[ele], requires_grad=False)

    return StreamingState(y)

################
#### TEST  ####
################

def test_running_and_final(streamed_data_obj, run_standards, final_standards):
    # Create spy
    test_logger = SpyLogger()

    avg_and_dev_obj = avg_and_dev(KEYS)
    avg_and_dev_obj.running(streamed_data_obj, test_logger)
    
    for ele in run_standards:
        for qty in run_standards[ele]:
            assert avg_and_dev_obj.state[ele][qty] == run_standards[ele][qty]

    avg_and_dev_obj.final(test_logger)
    
    for ele in KEYS:
        final_data = test_logger.write_json_calls[f"{ele}_dist_stats.json"]["dictionary"]
        for stat in final_data:
            assert final_standards[ele][stat] == pytest.approx(final_data[stat])
