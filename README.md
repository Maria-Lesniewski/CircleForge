# Doodle Generator 

This folder contains a small image generation project that uses generative modeling to "discover" parameters underlying images that are procedurally generated/generate images from the same family.

It contains XXXX Parts

### Part 1: Procedurally Generated Doodles (Training Data)
The script XXX.py generates a data base of doodles for the model to learn from. 

*Procedure Coded*:
The algorithm is a hardcoded version of some drawings I used to do while doodling in highschool. 
Large Circles fill the page randomly without intersecting (N circles of size M) -> Smaller circles fill the perimeter around those circles without touching it -> Lines create a negative space border between the circles and the remaining area -> Speckles fill the remaining area. 

*User Options/Input*:
The script is set up to play with and explore.
I like to think about how models perform when they have more / less data / more/less variability within that data. 
