# @file experiment_runner.py
# @author Maria Lesniewsk
# @desc This is a file for launching command line args on driver scripts in python

import subprocess
import sys

def run_job(cmd : list[str], inp : str=None):
    """
    Runs the string of args passed (as if they were CL words)
    inputs answers seperated by a \n for enter
    """
    print(f"\nRunning: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, input=inp, text=True)

    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}",flush=True)
        sys.exit(result.returncode)

    print("Finished successfully.", flush=True)


