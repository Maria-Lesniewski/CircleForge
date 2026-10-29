# @author Maria Lesniewski
# @file generate_training_data.py
# @desc This is the driver for creating procedurally generated circle drawings in a data base.
import sys
import importlib.util
import utils.io as io
import generator.circle_doodle as gen
from generator.default_config import DEFAULT_CONFIG
import numpy as np
from mpi4py import MPI
import pandas as pd
import json
from pathlib import Path

## THIS SCRIPT CAN BE RUN IN PARALELL

### MPI TRACKING
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

config = None
### Obtain Data Generation Parameters ###
if rank == 0:
    args = io.parse_config_args()
    if (args.config is not None):
        config = io.load_config_from_CL(args.config)
    else:
        config = DEFAULT_CONFIG

config = comm.bcast(config, root=0) 

negative_space_fill_densities = config["fill_frac"]
max_N_to_try = config["max_N_to_try"]
samples_per_setting = config["samples_per_setting"]
write_root = Path(config["data_root_filepath"])

### SANITY CHECK w. User
if size == 1:
    info = "This script generates alot of data\n" + "It can be run on N processors in parallel with\n" + "mpirun -np <N> python generate_training_data.py"
    q = "Are you sure you want to run on 1 processor?"
    sentinal = io.yes_no_q(extra_info=info, question=q)
    if (not sentinal):
        exit()
if rank == 0:
    info = ("Predicted settings will generate "+  
    f"{len(negative_space_fill_densities)*len(max_N_to_try)*samples_per_setting} samples\n"
    f"in path {io.artifact_path(write_root)}")
    q = "Is this okay?\n"
    sentinal = io.yes_no_q(extra_info=info, question=q)
    if (not sentinal): 
        print("Change Settings and try again")
        print(f"killing {size} processes")
        comm.Abort(0)

comm.Barrier()
### GENERATION STARTS HERE ###
#### Multi-data stream set up 
annotation_filenames = []

if rank == 0: # Make directories just once
    annotations_path = io.make_artifact_dir(write_root)
    sample_path = io.make_artifact_dir(write_root / "samples")

# but store info for writing across processor
annotations_path = io.artifact_path(write_root)
sample_path = io.artifact_path(write_root / "samples")

for i in range(0,size):
    annotation_filenames.append( annotations_path / f"processor_{i}_raw_annotations.csv")

# Generate Data, set seed for reproducibility
np.random.seed(rank)

job_ids = []
count = 0
for sample_no in range(0,samples_per_setting):
    for max_N in max_N_to_try:
        for pixel_density in negative_space_fill_densities:
            job_ids.append([count,max_N, sample_no, pixel_density])
            count+=1

for ele in job_ids:
    if (ele[0] % size ) == rank:
        writefilename = sample_path / f"Nmax_{ele[1]}_p{ele[3]}_{ele[2]}.npy"
        doodle_obj = gen.CircleDoodle(size=50, max_circle = ele[1], fill_percent = ele[3], filename=writefilename, annotations=annotation_filenames[rank])
        doodle_obj.fill_space()
        print(f"writing {writefilename}")
        doodle_obj.add_to_data_base(rootdir=None)

comm.Barrier()
# Rank 0 stitches files
if rank == 0:
    print("Combining CSVs...")
        
    combined = pd.concat(
        [pd.read_csv(f) for f in annotation_filenames],
        ignore_index=True
    )

    combined.to_csv(annotations_path  / "combined_raw_annotations.csv", index=False)

    print("Done")
