import pytest 
import sys 
from script_test_factory import ScriptTestRunner
from test_utils.toy_database import ToyDatabase
import utils.io as io
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


@pytest.fixture
def toy_database():
    """
    defines a toy data base for cpu-independent analysis of scripts that depend on other steps
    Settings mimick the config file passed to the test_Generate_training_data_w_config func
    """
    db = ToyDatabase(io.artifact_path("101test_experiment101"))
    db.fill_densities = [0.1]
    db.max_N = [1,2,3,4,5]
    db.seed = 0 # main script sets seed to processor rank
    db.samples_per_setting = 2
    db.generate()
    yield db # Normally we yield / clean, but the ct fixture handles cleaning



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

def test_scrub_split_data_w_config(test_artifact_dir, ct, toy_database):
    target_files = [
            "samples_removed_from_raw.csv",
            "train_val_test_split_annotations.csv",
            ]
    standard_files = []
    output_files = []
    for ele in target_files:
        standard_files.append("fixtures/standards/results/final_data/"+f"{ele}")
        output_files.append(f"{test_artifact_dir}/final_data/"+f"{ele}")

    ScriptTestRunner.test_script(
            driver_script="scrub_and_split_data.py",
            input_files={ "config" : "test_scripts/fixtures/inputs/data_split_config.py"},
            output_files=output_files,
            standard_files=standard_files,
            )
