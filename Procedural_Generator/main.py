# @author Maria Lesniewski
# @file main.py
# @desc This is the driver for my silly physics based diffusion experiment

import generatorlib as gen
import matplotlib.pyplot as plt
import numpy as np


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
q = doodle_obj.label_datapoints_with_area()
print(q)
plt.title( str(doodle_obj.nsmall) + " " +  str(doodle_obj.nbig) + " " + str(doodle_obj.ncircles))
plt.imshow(doodle_obj.image, cmap='gray')
plt.axis('off')
plt.show()
