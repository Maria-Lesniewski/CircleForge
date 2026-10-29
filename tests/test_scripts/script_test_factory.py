import utils.io as io
import sys
from launcher.experiment_runner import run_job
import json 
import pytest
from pandas.testing import assert_frame_equal
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
SCRIPTS = io.find_project_root() / "scripts"
HERE = io.find_here() 
STANDARDS = HERE / "standards" / "results"
# PASS PATHS RELATIVE TO THIS FILE OR THE ARTIFACT DIR 
class ScriptTestRunner:

    """
    Checks that each script in the scripts directory produces the expected output
    given fixed config file
    """

    @classmethod
    def compare_values(cls, a, e, path="$", tol : float=1e-12 ) -> None:
        if isinstance(e, dict):
            assert isinstance(a, dict), (
                f"{path}: expected a JSON object"
            )

            assert a.keys() == e.keys(), (
                f"{path}: JSON keys differ. "
                f"Actual={list(a.keys())}, "
                f"Expected={list(e.keys())}"
            )

            for key in e:
                if "path" not in key: # We don't want to check dictionary keys containing CPU paths
                    ScriptTestRunner.compare_values(
                        a[key],
                        e[key],
                        f"{path}.{key}",
                        tol,
                    )

        elif isinstance(e, list):
            assert isinstance(a, list), (
                f"{path}: expected a JSON array"
            )

            assert len(a) == len(e), (
                f"{path}: array lengths differ. "
                f"Actual={len(a)}, Expected={len(e)}"
            )

            for i, (av, ev) in enumerate(zip(a, e)):
                ScriptTestRunner.compare_values(av, ev, f"{path}[{i}]", tol)

        elif (
            isinstance(e, (int, float))
            and not isinstance(e, bool)
        ):
            assert isinstance(a, (int, float)), (
                f"{path}: expected a number, got {type(a).__name__}"
            )

            assert not isinstance(a, bool), (
                f"{path}: expected a number, got bool"
            )

            assert abs(a - e) <= tol, (
                f"{path}: numeric values differ. "
                f"Actual={a}, Expected={e}"
            )

        else:
            assert type(a) is type(e) and a == e, (
                f"{path}: values differ. "
                f"Actual={a!r}, Expected={e!r}"
            )

    @classmethod
    def compare_csv(cls, actual: Path, expected: Path, tol : float=1e-12) -> None:
        """Compare two CSV files by their parsed contents."""

        empty_act = False
        empty_exp = False
        actual_df = None
        expected_df = None

        try:
            actual_df = pd.read_csv(actual).drop(columns=["access_filename", "image_filename"], errors="ignore")
        except pd.errors.EmptyDataError:
            empty_act = True
            actual_df = pd.DataFrame({"Dummy" : [0]}) # Provide dummy value to compare if both are empty

        try:
            expected_df = pd.read_csv(expected).drop(columns=["access_filename", "image_filename"], errors="ignore")

        except pd.errors.EmptyDataError:
            empty_exp = True
            expected_df = pd.DataFrame({"Dummy" : [0]})
        assert empty_exp == empty_act
        
        try: 
            assert_frame_equal(
                actual_df,
                expected_df,
                check_dtype=True,
                check_exact=True,
                check_like=False,)
        except AssertionError as exc:
            raise AssertionError(
                    f"CSV files differ!\n" + 
                    f"Actual: {actual}\n" +
                    f"Expected: {expected}\n" +
                    f"Details {exc}" ) from exc

    @classmethod
    def compare_json(cls, actual: Path, expected: Path, tol : float=1e-12) -> None:
        """
        Compare JSON structures recursively.

        Numeric values may differ within the specified tolerances.
        Object keys, array lengths, strings, booleans, and null values must match.
        """

        with actual.open(encoding="utf-8") as f:
            actual_data = json.load(f)

        with expected.open(encoding="utf-8") as f:
            expected_data = json.load(f)

        ScriptTestRunner.compare_values(actual_data, expected_data, tol=tol)

    @classmethod
    def compare_xvg(cls, actual : Path, expected: Path, tol : float =1e-12) -> None:
        """
        Compares each value within an .xvg file loaded (similar to comparing a csv)
        w. different delimiting behavior and tol specified
        """
        actual_df = pd.read_csv(actual, sep=r"\s+", header=None,comment="#",)

        expected_df = pd.read_csv(expected, sep=r"\s+", header=None, comment="#")
        assert_frame_equal(actual_df,expected_df,check_dtype=False,check_exact=False, atol=tol)


    @classmethod 
    def compare_npy(cls, actual : Path or str, expected : Path or str, tol : float=1e-12) -> None:
        """
        Compares .npy files and asserts equal
        """
        a = np.load(actual)
        e = np.load(expected)
        assert np.array_equal(a,e)

    @classmethod
    def test_filetype(cls,act, exp, tol):
        str1_filetype = act.split(".")[-1]
        str2_filetype = exp.split(".")[-1]

        if str1_filetype != str2_filetype:
            assert False
        if str1_filetype == "json":
            ScriptTestRunner.compare_json(HERE/Path(exp), io.artifact_path(act), tol)
        elif str1_filetype == "csv":
            ScriptTestRunner.compare_csv(HERE/Path(exp), io.artifact_path(act), tol)
        elif str1_filetype == "xvg":
            ScriptTestRunner.compare_xvg(HERE/Path(exp), io.artifact_path(act), tol)
        elif str1_filetype == "npy":
            ScriptTestRunner.compare_npy(HERE/Path(exp), io.artifact_path(act), tol)
        else:
            raise ValueError("Filetype found not one of the expected types")


    @classmethod
    def test_script(cls, driver_script : str,input_files : dict[str:str], output_files : list[str], standard_files : list[str], inp : str = None, tol=1e-12):
        model_str = None
        config_str = None
        if ("model" in input_files.keys()):
            model_str = input_files["model"]
        config_str = input_files["config"]
        
        if model_str is None:
            CL_args = [str(SCRIPTS / driver_script), "--config", config_str ]
        else:
            CL_args = [str(SCRIPTS / driver_script), "--config", config_str, "--model" , model_str]
        cmd = [sys.executable, *CL_args]
        run_job(cmd, inp=inp)

        if len(output_files) != len(standard_files):
            raise ValueError("Same number of expected files and standard files must be passed")
        for actual, expected in zip(output_files, standard_files):
            if not io.artifact_path(actual).exists():
                print(f"{io.artifact_path(actual)} does not exist")
                assert io.artifact_path(actual).exists()
            assert io.artifact_path(actual).exists()

            ScriptTestRunner.test_filetype(actual, expected, tol)

    @classmethod
    def clean_script_test(cls):
         # Clean up artifacts (using a weird name for now to avoid overwriting someone's test dir)
        expected_output_path = io.artifact_path("101test_experiment101")
        if (expected_output_path.exists()):
            shutil.rmtree(expected_output_path)

