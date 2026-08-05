# @author Maria Lesniewski
# @file main.py
# @desc This is the driver for creating procedurally generated circle drawings in a data base.

import generatorlib as gen
import numpy as np
import os
from mpi4py import MPI



"""
### Data Generation Parameters 
negative_space_fill_densities = [0.1, 0.01, 0.5, 1]
max_N_to_try = list(range(0, 10))
max_N_to_try.extend(range(20, 400, 10))
samples_per_setting = 50
"""
negative_space_fill_densities = [1]
max_N_to_try = list(range(0, 3))
samples_per_setting = 4


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    while(True):
        print(f"Predicted settings will generate {len(negative_space_fill_densities)*len(max_N_to_try)*samples_per_setting} samples")
        print("Is this okay?\n")
        response = input("Y/N")
        if response == "Y":
            break
        if response == "N":
            print("Change Settings and try again")
            print(f"killing {size} processes")
            comm.Abort(0)
        else:
            print("Please input 'Y' or 'N'")

comm.Barrier()

print(f"I am rank {rank} out of {size}")
annotation_filenames = []
for i in range(0,size):
    annotation_filenames.append(f"processor_{i}_annotations.csv")

print(annotation_filenames)
# Generate Data, set seed for reproducibility
np.random.seed(rank)

if rank == 0:
    file_exists = os.path.exists("data_folder")
    if not file_exists:
        os.mkdir("data_folder")

job_ids = []
count = 0
for sample_no in range(0,samples_per_setting):
    for max_N in max_N_to_try:
        for pixel_density in negative_space_fill_densities:
            job_ids.append([count,max_N, sample_no, pixel_density])
            count+=1

for ele in job_ids:
    if (ele[0] % size ) == rank:
        writefilename = f"Nmax_{ele[1]}_p{ele[3]}_{ele[2]}.npy"
        doodle_obj = gen.CircleDoodle(size=600, max_circle = max_N, fill_percent = 0.01, filename=writefilename, annotations=annotation_filenames[rank])
        doodle_obj.fill_space()
        print(f"writing {writefilename}")
        doodle_obj.add_to_data_base("data_folder")
