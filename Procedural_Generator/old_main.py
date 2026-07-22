# @author Maria Lesniewski
# @file driver.py
# @desc This is the driver for my silly physics based diffusion experiment

import generatorlib as gen
import matplotlib.pyplot as plt



img = gen.generate_circle_doodle(300)
#img = gen.generate_circle_doodle(256, 6)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()