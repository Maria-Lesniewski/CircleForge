# Experiment 2: Feature Engineering
This is a follow up to the first experiment.
- In the first experiment we directly solve for the parameters of a linear model in an underdetermined set of data / model parameters. 
	- We saw that models essentially memorized their own training data, while doing poorly outside the training conditions.

- In this experiment, we examine the overdetermined regime, where the model has fewer parameters than there are examples to learn from. 
	- We ask if the model still performs as well on the training data in this overdetermined regime and if simple priors on the data can improve it's transferability. 

# To Answer This Question
We build linear models using direct linear regression on increasingly varied databases

## Procedure:
- **Dataset**	

	- We generate a single LARGE data set of 50k samples. 

		- 200 hidden max circles variable, 5 noise pixel densities, 50 samples per setting (~30k training samples)
			- Data sets are generated with hidden parameters then shuffled and randomly split to a training (%60), validation (%20) and test (%20) data set for each letter. 

- **Models**
	
	- Train a linear model, $(N = \bold{W}\bold{x} + b)$, on the data set to predict the number of circles in an image $N$ as a function of the raw pixel values $\bold{x}$. We train it using direct linear least squares fitting.
		- We refer to this model as the "BasicLinear" model
	
	- Train a linear model, $(N = \bold{W}\hat{\bold{x}} + b)$, on the data set to predict the number of circles in an image, $N$, as a function of the rotation manipulated pixel value $\hat{\bold{x}}$. We train it using direct linear least squares fitting. 
		- $\hat{\bold{x}}$ is the average of the pixel values when the image is rotated by 0, 90, 180, and 270 degrees. The represents a rotation invariant feature engineering on the representation of pixel values. (The idea is that the number of circles in an image should be the same when we rotate it)

		- We refer to this model as the "RotatedLinear" model

	- Train a constant model $N = c$ using least squares. 
		- Linear least squares on a constant model produces a "mean predictor model", we use this model to compare if the linear models are beating the naive most guess based on statistics. 
		- We refer to this model as the "Constant" model. 

- **Evaluation**

	- As in experiment 1, we measure the **$R^2$**, **accuracy** and **RMSD** of each model on the training, validation and test splits. 

### Quickstart? 
`bash end_to_end.sh`
- The data generated for this experiment is somewhat CPU intensive (~40 minutes on 4 cores using up to 8GB of ram end to end),
	-  Evaluate your resources before quickstarting. 
	- Consider running each pipeline script manually via `python run_pipeline<X>.py` or by calling each submodule on the CL  `python scripts/<script> --config <config>` as appropriate.

### Results
- Each stage of analysis will be stored in `artifacts/feature_engineering_experiment`. 
- A summary will be printed with each model's of performance. We provide an example of pre-obtained values in this folder.

## Considerations:

- Images are 50x50, since we're predicting the number of circles we see in an image that implies the model has (2,500 inputs->1 output). Counting the bias parameter, we have 2,501 model parameter (significantly fewer than the 30k training samples) so this data set represents an over determined system.

- Linear models are trained in double precision directly with least squares on CPU (default torch linalg solver will be poorly conditioned, script calls a more expensive one) before they're cast to torch.float32. (expect some numerical noise in casting)

- Since models are trained with direct least squares on the training set and no further consideration, the validation and test sets are each effectively a held out data set with similar distribution.

- Each set has a constant 50 samples per (max N, noise pixel density) variable employed

# Hypotheses/Expected Yields:
H : 
- The pixel densities in this data set should be less correlated overall with the circle count than we saw in experiment 1. (due to more noise combinations) 

- We expect that that RotatedLinear model may fit the training data worse than the BasicLinear model. (i.e. increased RMSD). We expect this because the feature rotation could act as a constraint on the solution. However, we hope this feature choice yields a better transfering model (better validation/test scores). 

- Conversely, because it lives in a less constrained pixel space, we expect that the BasicLinear model may fit the training data better, and transfer to other data sets worse. 

- Since data splits are randomly generated from shuffling samples, we expect each distribution to be similar, thus we expect the Constant model will be about consistent across data splits. 

- We expect each linear model will describe the variation in each data set better than the Constant (by near definition of $R^2$). 

EY: 

- All models are least squared optimized and should thus have R2 between 0 and 1 for the training data (with the Constant (mean predictor) model at 0)

- The Constant model should result in the average N\_circles from the training set
