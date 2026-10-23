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
		d) Two hidden max circles variables, various noise densities (like b) (~500 training samples)
		- Data sets are generated with hidden parameters then shuffled and randomly split to a training (%60), validation (%20) and test (%20) data set for each letter. 

	- Train a linear model on each data set using direct least squares fitting

	- Evaluate each model on its native data set and the other three (a) (b) (c) (d) 

	### Measure 
		- The R2 of each model on each set (variance accounted for by the model)
		- The accuracy (within 1 of total circle count) of each model on each set (how many samples basically correct)
		- The RMSD ( avg magnitude of error) 

	### Quickstart? 
	- bash end\_to\_end.sh found in this folder

	### Where?
	- Each stage of analysis will be stored in artifacts/intro\_experiment. 
	- A summary will be printed with each matrix of performance. Example in this folder.


## Considerations:
	- Images are 50x50, we're predicting the number of circles we see in an image with  (2500 inouts->1 output ), data sets contain less than 2501 samples, so the solution weights won't be unique
	- Linear models are trained in single precision directly with least squares on CPU (default torch linalg solver will be poorly conditioned, script calls a more expensive one)
	- Each set has a constant 250 samples per max N variable employed
	- Set A/C share a common NMax parameter and background noise conditions
	- Set B/D share a common NMax parameter and background noise conditions  


# Hypotheses/Expected Yields:
	- We should see that each linear model can effectively memorize their training set 
	  (R2 =1, accuracy = 100%, RMSD very small) 
	- We expect that each model will transfer poorly to conditions outside their training set
		- Sharp drop off in performance between training / validation + test
		- Sharper drop off for different sets
	- Particularly, linear models are taking advantage of a direct correlation between the amount of non-zero pixels and the number of circles counted, so the speckled drawings we expect will perform worse.
		-i.e. set_b / set_d models have steeper drop off between train & val/test

	- Set A/C and B/D may transfer "the best" to their respective pair, but we expect the two hidden Nmax variable states (e.g. "D") to transfer to their "one" partner (e.g. "B") than the "one
 partner does to the "two". 
