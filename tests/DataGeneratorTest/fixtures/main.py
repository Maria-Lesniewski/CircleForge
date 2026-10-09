# @author Maria Lesniewski
# @file main.py
# @desc This is a driver for making and comparing doodle images

import circle_doodle as gen
import numpy as np
import os
import random
import sys 

# Generate Data 
# Drawings have some characteristics like 
# Number of circles
# Area of blank negative space 
# 
# Generate a blank canvas (all black, 0. White = 255)
#doodle_obj = gen.CircleDoodle(600)
#doodle_obj = gen.CircleDoodle(300)
filepath = os.path.dirname(os.path.abspath(__file__))
doodle_obj = gen.CircleDoodle(size=600, max_circle = 1, fill_percent = 0.01, filename="name.npy")

# Fill the page with circles and the negative space with a random density of speckles
doodle_obj.fill_space()
doodle_obj.show_obj()

file_exists = os.path.exists("data_folder")
if not file_exists:
    os.mkdir("data_folder")
doodle_obj.add_to_data_base("data_folder")


doodle_obj2 = gen.CircleDoodle(size=600)
doodle_obj2.load_circle_obj_from_data("data_folder/annotations.csv","data_folder/name.npy")
doodle_obj2.show_obj()

os.remove("data_folder/annotations.csv")
os.remove("data_folder/name.npy")
os.rmdir("data_folder")


if (doodle_obj.is_equivalent(doodle_obj2)):
    print("The two objects are equivalent")
else:
    print("NOPE")
input("Press enter to end program")

os.mkdir("fixtures")
ones_path = str(os.path.join(filepath, "ones.npy"))
doodle_obj3 = gen.CircleDoodle(filename=ones_path)
doodle_obj3.image = np.ones((doodle_obj3.size, doodle_obj3.size))
doodle_obj3.add_to_data_base(".")

random.seed(67)
np.random.seed(67)
fpath = str(os.path.join(filepath, "fixture.npy"))
doodle_obj4 = gen.CircleDoodle(size=400, filename="fixtures/fixture.npy")
doodle_obj4.fill_space()
doodle_obj4.show_obj()
input("hey")
doodle_obj4.add_to_data_base(".")

