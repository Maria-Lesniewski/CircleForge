# This experiment is the last part of the intro experiment series (part 3)
- In this experiment we use gradient descent via the Adam optimizer to train a model with more complex architecture and measure its performance 

RECAP:
- In the intro, we directly solved for the parameters of a linear model with more parameters than observables to observe some potential pitfalls in overfitting data. (The models were great!...at their own training data, performance sharply fell off between data sets)

- In the part 2 experiment, we moved to an over determined regime, and found the linear model no longer sufficed for even its own training data, while still being unreliable outside of it. We explored some potential feature engineering to help the model generalize better, but ultimately found it insufficient.

# Question: 
- In this experiment, we add some more complexity to the models (a convolutional layer and nonlinear activation), and learn to train with gradient descent. We hope to see that the added complexity improves the model's performance


## Procedure 
	### Experiment

	**Data**
	- We repeat the data set used in [experiment 2: Feature Engineering](../feature_engineering_experiment/README.md)
	
	**Models**

	- We use an evolution of the RotatedLinear model:
		- (90 degree rotationally invariant data rep)
		- 3 Convolution filters with 5x5 kernels feed into a RELU layer before connecting to the standard linear network:
			- 25x3 kernal parameters = 75
			- 3 linear layers on 50 px images = 50x50x3 = 7500 
			This model : a total of 7,576 parameters

	**Training**:
	We employ 100 epochs (batch size 512) with the adam optimizer with a learning rate under the maximum theoretical optimization rate as determined from the linear model. To avoid overfitting, we only save models that improve the validation set while training on the training split. (best_checkpoint_epochs)

	**Measures**
	
	- R2, RMSD, and accuracy across sets 

# Hypotheses/Expected Yields: 

	EY:
	- We expect the loss to converge quickly within 100 epochs, to a lower loss threshold than either of the linear models previously checked.

	H: 
	- The added complexity will improve the model across training, validation and test. 

# Quickstart?
	- bash end\_to\_end.sh, results will be stored in artifacts/nn\_experiment

# Bonus:
	- One confounding factor in comparing linear models to the convolutional layer with activation is that we've tripled the parameters including in the model by connecting the three channels of convolutional filter to the linear (fully connected) layer. 
	- CircleForge implements a version of the RotatedLinear model with just the convolutional layer (no activation! Totally linear in principle). We have found that gradient descent on this model is pretty slow. We leave it as an exercise to the user to modify this pipeline to find out themselves. 

