# This Experiment Asks: 
Can linear regression use pixel inputs to determine the number of circles in the cartoon picture reliably? 

# To Answer This Question
We build linear models using direct linear regression on increasingly varied databases

## Procedure:
	### Experiment

	- Generate 4 data sets:
		a) One hidden max circles variable, no noisey pixel placements (~250 training samples)
		b) One hidden max circles variable, various noise densities (~250 training samples)
		c) Two hidden max circles variables, no noisey pixel placements (~500 training samples)
		d) Two hidden max circles variables (~500 training samples)

	- Train a linear model on each data set using direct least squares fitting

	- Evaluate each model on its native data set and the other three (a) (b) (c) (d) 

	### Measure 
		- The R2 of each model on each set (variance accounted for by the model)
		- The accuracy (within 1 of total circle count) of each model on each set (how many samples basically correct)
		- The RMSD ( avg magnitude of error) 

	### Quickstart? 
	- Run pipelines 1-3 found in this folder 

# Hypothesis:
	We should see that a linear model can memorize a set of training data, but that it willtransfer poorly to other sets. Transferability should get worse as the direct correlation between the ammount of non-zero pixels and the number of circles we count gets worse. 


## Considerations:
	- Images are 50x50, we're predicting the number of circles we see in an image with  (2500 inouts->1 output ), data sets contain less than 2501 samples, so the solution weights won't be unique
	- Linear models are trained in single precision directly with least squares on CPU (default torch linalg solver will be poorly conditioned, script calls a more expensive one)
	- Each set has a constant 250 samples per max N variable employed  
