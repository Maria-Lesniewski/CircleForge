# @author Maria Lesniewski
# @file main.py
# @desc This is the driver for my silly physics based diffusion experiment

import generatorlib as gen
import numpy as np
import os


# Generate Data 
# Drawings have some characteristics like 
# Number of circles
# Area of blank negative space 
# 
# Generate a blank canvas (all black, 0. White = 255)
#doodle_obj = gen.CircleDoodle(600)
#doodle_obj = gen.CircleDoodle(300)
doodle_obj = gen.CircleDoodle(size=600, max_circle = 1, fill_percent = 0.01)

# Fill the page with circles and the negative space with a random density of speckles
doodle_obj.fill_space()
doodle_obj.label_datapoint_with_ncircles()

doodle_obj.show_obj()

file_exists = os.path.exists("data_folder")
if not file_exists:
    os.mkdir("data_folder")
doodle_obj.add_to_data_base("data_folder")


doodle_obj2 = gen.CircleDoodle(size=600)
doodle_obj2.load_circle_obj_from_data("data_folder/annotations.csv","data_folder/default.npy")
doodle_obj2.show_obj()

os.remove("data_folder/annotations.csv")
os.remove("data_folder/default.npy")
os.rmdir("data_folder")

input("Press enter to end program")
