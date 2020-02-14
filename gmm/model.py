################################################# defining G and D #############################################
import torch
from torch import nn
import torch.nn.functional as F


# start writing networks

# discriminator: this is a binary classifier,
class DNet(nn.Module):
    '''four-layer MLP'''
    def __init__(self):
        # call the initializer of the superclass (base class)
        super(DNet, self).__init__()
        n_features = 2   # input dimension
        n_out = 1       # output dimension
        self.ac = F.relu
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.ac(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# generator: Z (noise, usually Gaussian) -> X (images)
class GNet(nn.Module):
    '''four-layer MLP'''
    def __init__(self):
        # call the initializer of the superclass (base class)
        super(GNet, self).__init__()
        n_features = 100  # generate from a hidden distribution
        n_out = 2  # the dimension of the point
        self.ac = F.relu
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)     # we are generating 2d Gaussian

    def forward(self, x):
        x = self.ac(self.fc1(x))
        x = self.ac(self.fc2(x))
        x = self.ac(self.fc3(x))
        x = self.fc4(x)
        return x

