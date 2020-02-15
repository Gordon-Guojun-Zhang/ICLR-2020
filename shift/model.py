import torch
import numpy as np

################################################# defining G and D #############################################

# functions

def dis(theta, x):      # compute (1/N)sum_{i=1}^N s(theta.x[i])
    size = x.shape[0]   # how many samples
    total = torch.tensor(0)
    for _ in range(size):
       total = total + torch.sigmoid(torch.dot(theta.float(), x[_]))
    return total / size

def gen(phi, x):
    return torch.add(phi, x)
