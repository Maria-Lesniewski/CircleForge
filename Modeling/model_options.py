import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

""" TO DO 
class PoissonLinear(nn.Module):

criterion = nn.PoissonNLLLoss(log_input=False)

rate = model(x)
loss = criterion(rate, y.float().unsqueeze(1))
""" 
