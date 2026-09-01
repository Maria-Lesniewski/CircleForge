import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicLinear(nn.Module):
    def __init__(self):
        super(BasicLinear,self).__init__()
        self.flatten = nn.Flatten() # We flatten from dim 1 on so that batches are handled
        self.f1 = nn.Linear(50*50, 1) # linear layers are called fully connected
        # Note that nn.Linear has bias=true by default so we have 50^2 + 1 params 

    def forward(self, x):
        x = self.flatten(x)
        x = self.f1(x)
        return x


""" TO DO 
class PoissonLinear(nn.Module):

criterion = nn.PoissonNLLLoss(log_input=False)

rate = model(x)
loss = criterion(rate, y.float().unsqueeze(1))
""" 
