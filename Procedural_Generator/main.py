# @author Maria Lesniewski
# @file main.py
# @desc This is the driver for my silly physics based diffusion experiment

import generatorlib as gen
import matplotlib.pyplot as plt
import numpy as np

# Generate a blank canvas (all black, 0. White = 255)
doodle_obj = gen.CircleDoodle(600)

# Fill the page with circles and the negative space with a random density of speckles
doodle_obj.fill_space()

plt.imshow(doodle_obj.image, cmap='gray')
plt.axis('off')
plt.show()