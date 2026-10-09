# @author Maria Lesniewski
# @file generate_training_data.py
# @desc This is the driver for creating procedurally generated circle drawings in a data base.
import os
import utils.io as io
import ProceduralGenerator.circle_doodle as gen
import numpy as np
from mpi4py import MPI
import pandas as pd

### Data Generation Parameters 
negative_space_fill_densities = [0.1, 0.01, 0.3, 0.5, 1]
max_N_to_try = list(range(0,200))
#np.array([1]) # list(range(0, 200))
samples_per_setting = 50
#1 # 50

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size == 1:
    info = "This script generates alot of data\n" + "It can be run on N processors in parallel with\n" + "mpirun -np <N> python generate_training_data.py"
    q = "Are you sure you want to run on 1 processor?"
    sentinal = io.yes_no_q(extra_info=info, question=q)
    if (not sentinal):
        exit()
if rank == 0:
    info = f"Predicted settings will generate {len(negative_space_fill_densities)*len(max_N_to_try)*samples_per_setting} samples\n"
    q = "Is this okay?\n"
    sentinal = io.yes_no_q(extra_info=info, question=q)
    if (not sentinal): 
        print("Change Settings and try again")
        print(f"killing {size} processes")
        comm.Abort(0)

comm.Barrier()

annotation_filenames = []

print(rank)
if rank == 0:
    io.make_artifact_dir("raw_data")
    io.make_artifact_dir("raw_data/samples")

for i in range(0,size):
    annotation_filenames.append(io.artifact_path(f"raw_data/processor_{i}_raw_annotations.csv"))

print(annotation_filenames)

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
        writefilename = io.artifact_path(f"raw_data/samples/Nmax_{ele[1]}_p{ele[3]}_{ele[2]}.npy")
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

    combined.to_csv(io.artifact_path("raw_data/combined_raw_annotations.csv"), index=False)

    print("Done")
