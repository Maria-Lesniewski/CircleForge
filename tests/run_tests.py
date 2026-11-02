import os
import subprocess
import sys

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.exit(
    subprocess.call([
        sys.executable,
        "-m",
        "pytest",
        *sys.argv[1:]
    ])
)

