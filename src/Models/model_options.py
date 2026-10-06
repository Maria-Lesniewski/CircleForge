import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List

## Model Factory for writing generic test / train functions without cluttering code with too many variants that do the same thing
# The registry stores the list of models implemented within this module
class ModelFactory:
    _registry: Dict[str,Any] = {
            # We fill this in as we add types below
            }

    # Add to registry with decorators 
    @classmethod
    def register(cls, name : str):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    # Aggressive type hinting, this is a factory method
    def create(cls, model_type: str, config: Dict[str, Any]) -> nn.Module:
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: '{model_type}'")
        return cls._registry[model_type](**config) # config will be used as unpacked args for inits where applicable

@ModelFactory.register("BasicLinear")
class BasicLinear(nn.Module):
    """
    The most naive model we could build. 
    Simply 
    N = Wx, where we learn W via adam or LSTQs, N is the number of circles, and x is a (size*size,1) vector of pixel values 
    """
    def __init__(self):
        super(BasicLinear,self).__init__()
        self.flatten = nn.Flatten() # We flatten from dim 1 on so that batches are handled
        self.f1 = nn.Linear(50*50, 1) # linear layers are called fully connected
        # Note that nn.Linear has bias=true by default so we have 50^2 + 1 params 

    def forward(self, x):
        x = self.flatten(x)
        x = self.f1(x)
        return x
@ModelFactory.register("RotatedLinear")
class RotatedLinear(nn.Module):
    """
    A slightly fancier version of the naive model to account for count invariance under rotation
    The simple model is 
    N = Wx, 
    We want for any 90 degree rotation (and any such transform that leaves all pixels/circles on the screen), P
    A simple way to do this is to adjust the vector representation then operate
    """
    def __init__(self):
        super(RotatedLinear,self).__init__()
        self.flatten = nn.Flatten() # We flatten from dim 1 on so that batches are handled
        self.f1 = nn.Linear(50*50, 1) # linear layers are called fully connected
        # Note that nn.Linear has bias=true by default so we have 50^2 + 1 params 

    def forward(self, x):
        x = 1/4*(x + torch.rot90(x, 1, dims=(1,2)) + torch.rot90(x, 2, dims=(1,2)) + torch.rot90(x,3, dims=(1,2)))
        x = self.flatten(x)
        x = self.f1(x)
        return x

@ModelFactory.register("Constant")
class Constant(nn.Module):
    """
    A model who's use is almost purely educational / sanity checking/ comparative analysis based
    Least squares objectives should return the mean of the data set
    takes 
    """

    def __init__(self):
        super(Constant, self).__init__()
        self.bias = nn.Parameter(torch.tensor(float(torch.rand(1))))

    def forward(self, x):
        return self.bias.expand(x.shape[0]) # Return constant across batch size


""" TO DO 
class PoissonLinear(nn.Module):

criterion = nn.PoissonNLLLoss(log_input=False)

rate = model(x)
loss = criterion(rate, y.float().unsqueeze(1))
""" 
