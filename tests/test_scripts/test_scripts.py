import pytest 
import sys 
from script_test_factory import ScriptTestRunner

@pytest.fixture
def test_artifact_dir():
    return "101test_experiment101"

@pytest.fixture
def ct():
    """
    Assists in test clean up (regardless of pass fail)
    to avoid accumulated files in the artifacts directory
    """
    yield None
    ScriptTestRunner.clean_script_test()

def test_generate_training_data_w_config(test_artifact_dir, ct):
    target_files = [
            "processor_0_raw_annotations.csv",
            "combined_raw_annotations.csv",
            "samples/Nmax_2_p0.1_0.npy",
            "samples/Nmax_3_p0.1_0.npy",
            "samples/Nmax_4_p0.1_0.npy",
            "samples/Nmax_5_p0.1_0.npy",
            "samples/Nmax_1_p0.1_1.npy",
            "samples/Nmax_2_p0.1_1.npy",
            "samples/Nmax_3_p0.1_1.npy",
            "samples/Nmax_4_p0.1_1.npy",
            "samples/Nmax_5_p0.1_1.npy",
            ]
    standard_files = []
    output_files = []
    for ele in target_files:
        standard_files.append("fixtures/standards/results/raw_samples/"+f"{ele}")
        output_files.append(f"{test_artifact_dir}/raw_samples/" + f"{ele}")
    
    ScriptTestRunner.test_script(
            driver_script="generate_training_data.py",
            input_files={ "config" : "test_scripts/fixtures/inputs/generator_config.py"},
            output_files=output_files,
            standard_files=standard_files,
            inp="Y\nY\n",
            )

