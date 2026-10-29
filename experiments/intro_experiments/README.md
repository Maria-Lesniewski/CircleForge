# Experiment 1: Introduction 
We start modeling simply by asking "can linear regression use pixel inputs to determine the number of circles in the cartoon picture reliably?"

## To Answer This Question
We build linear models using direct linear regression on increasingly varied databases. 

## Procedure:
	
- **DataSet**

	- Generate 4 data sets:
		a) One hidden max circles variable, no noisy pixel placements (~250 training samples)
		b) One hidden max circles variable, various noise densities (~250 training samples)
		c) Two hidden max circles variables, no noisy pixel placements (~500 training samples)
		d) Two hidden max circles variables, various noise densities (like b) (~500 training samples)
		
		- Data sets are generated with hidden parameters then shuffled and randomly split to a training (%60), validation (%20) and test (%20) data set for each letter. 

- **Models**

	- Train a linear model $(N = \bold{W}\bold{x} + b)$ on each data set using direct least squares fitting to predict the number of circles in an image $N$, based upon the pixel representation of the image loaded, $\bold{x}$.

	- We refer to the model trained on "set a" as "model a", the model trained on "set b" as "model b", etc. 

- **Evaluation**
	- Evaluate each model on its native data set and the other three (a) (b) (c) (d) by measuring:
		- The **$R^2$** of each model on each set. We interpret this as the variance in the set that each model accounts for.
		- The **accuracy** of each model on each set.
			- This is a regression test, not a classification model yet, so we define accuracy as the fraction of inferences that are within 1 of the true circle count.
		
		- The **RMSD** of each model on each set, i.e. the average magnitude of inference errors. 
			- Since models are trained with least squares loss RMSD is equivalent to the square root of the loss function. (loss = Mean Square Error)  

### Quickstart? 
`bash end_to_end.sh`

### Results
- Each stage of analysis will be stored in `artifacts/intro_experiment`. 
- A summary will be printed with each matrix of performance. We provide an example of pre-obtained values in this folder.

## Considerations:
- Images are 50x50, since we're predicting the number of circles we see in an image that implies the model has (2,500 inputs->1 output). Counting the bias parameter, we have 2,501 model parameter (significantly more than the number of samples) so the solution weights won't be unique.

- Linear models are trained in double precision directly with least squares on CPU (default torch linalg solver will be poorly conditioned, script calls a more expensive one) before they're cast to torch.float32. (expect some numerical noise in casting)

- Each set has a constant 250 samples per max N variable employed

- "set a" and "set c" share a common NMax parameter and background noise conditions

- "set b" and "set d" share a common NMax parameter and background noise conditions  


# Hypotheses/Expected Yields:
- We have found experimentally that the condition number is pretty small for this experiment (order 1-10), we should see that each linear model can effectively memorize their training set (R2 =1, accuracy = 100%, RMSD very small) 

- We expect that each model will transfer poorly to conditions outside their training set
	- Sharp drop off in performance between training / validation + test
	- Sharper drop off for different sets

- Particularly, linear models are taking advantage of a direct correlation between the amount of non-zero pixels and the number of circles counted, so the speckled drawings we expect will perform worse.
	-i.e. set_b / set_d models have steeper drop off between train & val/test

- Because of each set's similarity, we expect "model a" to transfer relatively well "set c" and also "model b" to "set d".  However, we expect models trained on sets with two hidden Nmax variable states to their transfer to the subset partner more effectively. (e.g. "model d" transfers to "set b" better than "model b" to "set d")