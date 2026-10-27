# This Experiment Continues the intro experiment as Part 2 in a 3 part series
- In the first experiment we directly solve for the parameters of a linear model in an underdetermined set of data / model parameters. 
	- We saw that models essentially memorized their own training data, while doing poorly outside the training conditions
- In this experiment, we examine the overdetermined regime, where the model has fewer parameters than their are examples to learn from. 
	- We examine if the model still performs as well on the training data / if simple priors on the data can improve it. 

# To Answer This Question
We build linear models using direct linear regression on increasingly varied databases

## Procedure:
	### Experiment

	- Generate 1 LARGE data set:
		a) 200 hidden max circles variable, 5 noise pixel densities, 50 samples per setting (~30k training samples)
		- Data sets are generated with hidden parameters then shuffled and randomly split to a training (%60), validation (%20) and test (%20) data set for each letter. 

	- Train a linear model on the data set using direct least squares fitting using raw pixel values as input

	- Train a linear model on each data set using direct least squares fitting using a 90 degree invariant representation of pixel values as input
		- (This represents a prior that the number of circles in an image should be the same when we rotate it)

	- Train a constant model using least squares as a baseline to determine efficacy of model accuracy (a mean predictor model)

	### Measure 
		- The R2 of each model on each set (variance accounted for by the model)
		- The accuracy (within 1 of total circle count) of each model on each set (how many samples basically correct)
		- The RMSD ( avg magnitude of error) 

	### Quickstart? 
	- The data generated for this experiment is somewhat CPU intensive (~40 minutes on 4 cores, please evaluate your resources before quickstarting, consider running each pipeline script manually via python run\_pipelineX.py or by calling each submodule on the CL as appropriate followed by the collected analysis script.)
	- If your resources are sufficient, bash end\_to\_end.sh found in this folder [Review the NCores variable in run\_pipeline1.py to assure proper utilization]

	### Where?
	- Each stage of analysis will be stored in artifacts/feature\_engineering\_experiment. 
	- A summary will be printed with each model's of performance. Example in this folder.


## Considerations:
	- Images are 50x50, we're predicting the number of circles we see in an image with  (2500 inouts->1 output ), with 30k samples, we're in an over determined system
	- Linear models are trained in single precision directly with least squares on CPU with float64 precision before being cast to float32 for pytorch. 
		- (default torch linalg solver + single precision are poorly conditioned in this problem, script calls a more expensive one)
	- Since models are trained with direct least squares on the training set and no further consideration, the validation and test sets can each be considered different random splits of a test set.
	- Each set has a constant 50 samples per (max N, noise pixel density) variable employed

# Hypotheses/Expected Yields:
	H : 
	- The pixel densities in this data set should be less correlated overall with the circle count than we saw in experiment 1. (due to more noise combinations) 
	- We expect that the linear model may fit the training data better, and transfer to other data sets worse. 
	- We expect that that rotated linear model may fit the training data worse (RMSD ^ because rotation acts as a constraint on the solution), but we hope it transfers better
	- Since data splits are randomly generated from shuffling samples, we expect each distribution not to be significantly different, thus we expect the mean predictor model will be about 
	  consistent across splits. 
	- We expect that linear models will describe the variation in each data set better than the mean predictor. 
	
	EY: 
	- All models are least squared optimized and should thus have R2 between 0 and 1 for the training data (with the Constant (mean predictor) model at 0)
	- The Constant model should result in the average N\_circles from the training set
