# Model API
All models implement the PyTorch nn.Module interface and can be intstantiated directly or through [ModelFactory](model_factory.md). 

The models operate on image tensors representing images of size 50x50 

The images are assumed to be a batch of grayscale (black + white) images from the [CircleData](circle_data) class. 


## BasicLinear
<details>

<a id="BasicLinear"></a>

### <u>Purpose</u>

A baseline linear model for predicting a scalar quantity from an image

The model flattens each image into a vector and applies a fully connected linear transformation 

### <u>Architecture</u>

For an input image $\bold{x}$, the model flattens the image then computes:
 $$ \hat{y} = \bold{W} \bold{x} + b$$
 where: 

-  $\bold{x} \in \mathbb{R}^{2500}$ is the flattened image
- $\bold{W} \in \mathbb{R}^{1 \times 2500}$ is the learned weight matrix
- $b \in \mathbb{R}$ is the learned bias.
- $\hat{y} \in \mathbb{R}$  is the prediction.

**Input**

A tensor with shape $(B,50,50)$ where $B$ os the batch size. 

**Output**

A tensor with shape $(B,1)$ containing one scalar prediction per image. 

**Trainable Parameters**

The model contains $50\times50 + 1 = 2501$ parameters. 

### <u>Example</u>
```
# Create model
model = BasicLinear()

# Get a pretend image with batchsize 8
x = torch.randn(8, 50, 50) 

# 8 predicted scalars
predictions = model(x)
assert predictions.shape == (8,1)
```
</details>

## RotatedLinear
<details>

### <u>Purpose</u>
A linear model designed to make its predictions invariant to rotations by 90&deg;, 180&deg;, 270&deg;. 

It achieves this by averaging the input image over its four right-angle orientations before applying the same linear transformation used by [BasicLinear](#basiclinear)

### <u>Architecture</u>

Let $R_k(\bold{x})$ denote the image rotated by $k\times$ 90&deg;

The model first computes:

$$
\bold{x}_{\text{avg}} = \frac{1}{4} \sum_{k=0}^{3} {R_k(\bold{x})}
$$

It then flattens the averaged image and applies a linear transformation: 

 $$ \hat{y} = \bold{W} \bold{x}_{\text{avg}} + b$$
 where: 

-  $\bold{x}_{\text{avg}} \in \mathbb{R}^{2500}$ is the flattened rotationally averaged image
- $\bold{W} \in \mathbb{R}^{1 \times 2500}$ is the learned weight matrix
- $b \in \mathbb{R}$ is the learned bias.
- $\hat{y} \in \mathbb{R}$  is the prediction.

This makes the model invariant to the four right-angle rotations, assuming the rotations are implemented consistently and the input dimensions permit them.

**Input**

A tensor with shape $(B, 50, 50)$

**Output**

A tensor with shape $(B, 1)$

**Trainable Parameters**

The model contains 2,501 trainable parameters, identical to [BasicLinear](#basiclinear).

The rotation averaging operation introduces no trainable parameters

### <u>Example</u>
```
# Create model
model = BasicLinear()

# Get a pretend image with batchsize 8
x = torch.randn(8, 50, 50) 

# 8 predicted scalars
predictions = model(x)
assert predictions.shape == (8,1)
```

**Design Considerations**

The model aberages rotated images before applying the linear layer, this imposes a specific form of rotational invariance while keeping the model's parameter count unchanged. (A small feature engineering example)

</details>

## Constant
<details>

### <u>Purpose</u>

A simple baseline model that produces the same trainable scalar prediction for every input. 
This model is useful for sanity checks and comparisons agains more expressive models. 
For example, under a mean squared error objective, its optimal prediction on a fixed dataset is the mean of the target values. 

### <u>Architecture</u>

The model contains a single trainable scalar parameter $c$.

For every input sample $\bold{x}_i$, 
$$
    \hat{y}_i = c
$$

The input image does not affect the prediction.

**Input**

A batched tensor whose first dimension represents the batch size.

The model does not use the image contents

**Output**

A tensor with shape $(B,)$

Note that this differs from the output shape of the linear models, which is $(B,1)$.

**Trainable Parameters**

One trainable scalar parameter.


### <u>Example</u>
```
model = Constant()

# Data with some leading batch dimension
x = torch.randn(8,50,50)

# Predictions (8, ) (Other dimensions dont matter)
predictions = model(x)

assert predictions.shape == (8,)

```

**Design Considerations**

- The model's initial scalar value is randomly initialized. Training adjusts this parameter to minimize the chosen objective 
    - for mean squared error, the optimal scalar prediction is the mean of the training targets

    - for other objectives, the optimal value depends on the objective function 
</details>