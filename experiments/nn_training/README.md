# Experiment 3: nn_training
- In this final experiment we use gradient descent via the Adam optimizer to train a model with more complex architecture and measure its performance against previously generated linear models.

**Tutorial RECAP:**
- In the intro, we directly solved for the parameters of a linear model with more parameters than observables to observe some potential pitfalls in overfitting data. (The models were great!...at their own training data, performance sharply fell off between data sets)

- In the part 2 experiment, we moved to an over determined regime, and found the linear model no longer sufficed for even its own training data, while still being unreliable outside of it. We explored some potential feature engineering to help the model generalize better, but ultimately found it insufficient.

- In this experiment, we ask if a simple convolutional nueral network improves our previous regression performance. 

# To Answer This Question:
- We introduce a new model to Experiment 2: 
	- The RotatedConvolutionalReLU model is a function of normalized input pixel values $\boldsymbol{x}$ via:
	$$
	 N = \boldsymbol{W}{\boldsymbol{g}} + b \\
	 \boldsymbol{g} = \text{ReLU}(\boldsymbol{C}) = \max(\boldsymbol{0}, \boldsymbol{C}) \\
	 \boldsymbol{C} = \operatorname{Conv}_{5\times5}(\hat{\bold{x}})\\
	 \hat{\bold{x}}=    \frac{1}{4}\left(\boldsymbol{x} + R_{90^\circ}(\boldsymbol{x}) + R_{180^\circ}(\boldsymbol{x}) + R_{270^\circ}(\boldsymbol{x})\right) 
	$$
	where $\operatorname{Conv}_{5\times5}$ represents a convolutional layer with a $5\times5$ kernel and 3 output channels and $R_{X}$ represents a rotation of the image by $X^{\circ}$.
	- We normalize the pixel values by their maximum value (255) to help stabilize the gradient descent that we use during training


## Procedure 

**Data**
- We repeat the data set used in [experiment 2: Feature Engineering](../feature_engineering_experiment/README.md)

**Models/Training**

- We train the RotatedConvolutionalReLU model.
- We employ 100 epochs with batch size 512 using the adam optimizer and a learning rate of 0.002. 
- To avoid overfitting, we employ the "best_checkpoint_epochs" option, which only saves models that improve the validation set while training on the training split. 

**Measures**

- As in experiment 1 and 2, we measure the **$R^2$**, **accuracy** and **RMSD** of each model on the training, validation and test splits. 

# Considerations 

- The model employs 3 Convolution filters with 5x5 kernels, fed into a RELU layer before connecting to the standard linear network:

	Total Parameters
	- 25x3 kernel parameters = 75
	- 3 linear layers on 50 px images = 50x50x3 = 7500 
	This model : a total of 7,576 parameters.
	We are still under the number of samples and thus in an overdetermined regime. Although a 10:1 sample-to-parameter ratio is sometimes used as a rough rule of thumb, this is not a universal requirement for neural networks. We investigate whether this model can learn useful representations and generalize across the data distributions considered.

# Hypotheses/Expected Yields: 

EY:
- We expect the loss to converge quickly within 100 epochs, to a lower loss threshold than either of the linear models previously checked since the added parameters constrain the solution space less. 

H: 
- We hope that the added complexity will improve the model across training, validation and test. 

### Quickstart?

- `bash end_to_end.sh` : if you haven't completed part 2, there will be an error.
- input `training_config.py` assumes 4 worker available, adjust to your cpu resources.
- results will be stored in `artifacts/nn_experiment`

### Bonus Activities:

- **Closer Comparisons**
--  One would like to compare models with a similar number of parameters to truly say that the "activation" layer is helping the model explain the variance in the data. 
	- The confounding factor in comparing e.g. RotatedBasicLinear to RotatedConvolutionalReLU is that we've tripled the parameters including in the model by connecting the three channels of convolutional filter to the linear (fully connected) layer.

		**Test a more similar**
		- CircleForge implements a version of the RotatedLinear model with just the convolutional layer (no activation! Totally linear in principle). We have found that gradient descent on this model is pretty slow. We leave it as an exercise to the user to modify this pipeline to find out themselves. 

- The learning rate employed in this experiment was determined using `direct_linear_model_solver.py` to determine the conditioning of a linear problem, we use the hessian of this linear problem as a baseline for finding the maximum stable step size. The input for this is quite similar to that employed in experiment 2, where the feature_transform on $\bold{x}$ was `None`. In this problem, we scale $\bold{x}$ by $1/255$. We leave configuring direct_linear_model_solver for the scaled version of the problem as an easy excercise for the user. 