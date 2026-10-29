import pytest 
import sys 
from script_test_factory import ScriptTestRunner
from test_utils.toy_database import ToyDatabase
from launcher.experiment_runner import run_job
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

@pytest.fixture
def shuffled_db(toy_database):
    SCRIPTS = io.find_project_root() / "scripts"
    CL_args = [str(SCRIPTS/ "scrub_and_split_data.py"), "--config", "test_scripts/fixtures/inputs/data_split_config.py"]
    cmd = [sys.executable, *CL_args]
    run_job(cmd,inp=None)
    yield toy_database


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

def test_analyze_data_splits_w_config(test_artifact_dir, ct, shuffled_db):
    target_analysis = ["avg_and_dev", "correlate_feature", "correlate_labels", "histogram_labels"]
    an_files = {}
    an_files["avg_and_dev"] = ["nbig_dist_stats.json", "ncircles_dist_stats.json", "nsmall_dist_stats.json"]
    an_files["correlate_feature"] = [
            "nbig_vs_L2Norm_on_feature_nbig.xvg",
            "nbig_vs_LaplacianResponse_on_feature_nbig.xvg",
            "nbig_vs_mean_on_feature_nbig.xvg",
            "ncircles_vs_L2Norm_on_feature_ncircles.xvg",
            "ncircles_vs_LaplacianResponse_on_feature_ncircles.xvg",
            "ncircles_vs_mean_on_feature_ncircles.xvg",
            "nsmall_vs_L2Norm_on_feature_nsmall.xvg",
            "nsmall_vs_LaplacianResponse_on_feature_nsmall.xvg",
            "nsmall_vs_mean_on_feature_nsmall.xvg",
            ]
    an_files["correlate_labels"] = [
            "diagnostic_nbig_vs_ncircles.xvg",
            "diagnostic_nsmall_vs_nbig.xvg",
            "diagnostic_nsmall_vs_ncircles.xvg",
            ]
    an_files["histogram_labels"] = [
            "evaluation_nbig_histogram.xvg",
            "evaluation_ncircles_histogram.xvg",
            "evaluation_nsmall_histogram.xvg",
            ]

    target_files = []
    for split in ["test", "train", "val"]:
        for ele in target_analysis:
            for file in an_files[ele]:
                if ele != "avg_and_dev":
                    target_files.append(
                            f"DataDiagnostics/{split}/{ele}/grace/{file}")
                else:
                    target_files.append(
                            f"DataDiagnostics/{split}/{ele}/{file}")
    standard_files = []
    output_files = []
    for file in target_files:
        standard_files.append("fixtures/standards/results/" + f"{file}")
        output_files.append(f"{test_artifact_dir}/" + f"{file}")


    CL_string=""
    for i in range(0,3): # for each dataset analyzer
        CL_string+="0 1 2 3 -1\n" # Pick all options
        CL_string+="0 1 2 -1\n"
        CL_string+="10\n"# Configure the histogram - max_bin
        CL_string+="-10\n" # Configure the label histogram - min bin
        CL_string+="10\n" # configure the histogram - nbins

    ScriptTestRunner.test_script(
            driver_script="analyze_data_splits.py",
            input_files={ "config" : "test_scripts/fixtures/inputs/data_analysis_config.py"},
            output_files=output_files,
            standard_files=standard_files,
            inp=CL_string
            )

def test_direct_linear_model_solver_w_config_and_model(test_artifact_dir, ct, shuffled_db):
    target_files = [
            "spectral_analysis.json",
            "cpt_meta_data/BasicLinear_step_0.json",
            ]
    standard_files = []
    output_files = []
    for ele in target_files:
        standard_files.append("fixtures/standards/results/LeastSquaresResults/"+f"{ele}")
        output_files.append(f"{test_artifact_dir}/LeastSquaresResults/" + f"{ele}")

    ScriptTestRunner.test_script(
            driver_script="direct_linear_model_solver.py",
            input_files={ "config" : "test_scripts/fixtures/inputs/least_squares_config.py", 
                         "model" : "BasicLinear"
                         },
            output_files=output_files,
            standard_files=standard_files,
            inp=None,
            tol=1e-10
            )

def test_eval_models_w_config_and_CL_model(test_artifact_dir, ct, shuffled_db):
    """
    This is a minimal version of this test as I expand coverage and rework other structures
    """
    target_analysis = ["r2", "accuracy_within_tol", "rmsd"]
    an_files = {}
    an_files["r2"] = [
            "grace/evaluation_N.xvg",
            "grace/evaluation_R2.xvg",
            "grace/evaluation_SSE.xvg",
            "grace/evaluation_SST.xvg",
            "grace/evaluation_var.xvg",
            ]
    an_files["rmsd"] = ["grace/evaluation_RMSD.xvg",]
    an_files["accuracy_within_tol"] = ["accuracy.json",]
    
    target_files = []
    for split in ["train", "val", "test"]:
        for ele in target_analysis:
            for file in an_files[ele]:
                target_files.append(
                            f"ModelEvaluation/BasicLinear/{split}/{ele}/{file}")
    standard_files = []
    output_files = []
    for file in target_files:
        standard_files.append("fixtures/standards/results/" + f"{file}")
        output_files.append(f"{test_artifact_dir}/" + f"{file}")
    CL_string = str(io.find_here()/"fixtures/standards/results/LeastSquaresResults/checkpoints/BasicLinear_step_0.pt") + "\n"
    for i in range(0,3): # for each dataset analyzer
        CL_string+="0 4 5 -1\n" # 0: accuracy, 4:r2, 5: rmsd
        CL_string+="1\n" # Configure accuracy tolerance

    ScriptTestRunner.test_script(
            driver_script="analyze_models.py",
            input_files={ "config" : "test_scripts/fixtures/inputs/model_evaluation_config.py"},
            output_files=output_files,
            standard_files=standard_files,
            inp=CL_string,
            tol=1e-10
            )




## Still need a test_train_models, but device management and stochastic seeds need to be wrangled first. Some sample output is stored in the fixtures folder. 
