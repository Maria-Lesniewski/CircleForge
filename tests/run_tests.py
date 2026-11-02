import os
import subprocess
import sys

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"

import torch
print(torch.__config__.show())
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.exit(
    subprocess.call([
        sys.executable,
        "-m",
        "pytest",
        *sys.argv[1:]
    ])
)

