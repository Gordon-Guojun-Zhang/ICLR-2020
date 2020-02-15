import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from model import *


################################################# data preprocessing #############################################

X = np.load("one_gaussian.npy")
X = torch.tensor(X)
dim = 2
sample_size = X.shape[0]

# initialization, fix the same initialization
v = np.ones(dim)
np.random.seed(12)
phi = torch.tensor(v + np.random.randn(dim) * 0.1, requires_grad=True) 
np.random.seed(24)
theta = torch.tensor(np.random.randn(dim) * 0.1, requires_grad=True)
print("initial phi: ", phi)

################################################# optimizers #############################################
lr = 0.002
r1 = 0.9
r2 = 0.999
d_opt = optim.Adam([theta], betas=[r1, r2], lr=lr)   
g_opt = optim.Adam([phi], betas=[r1, r2], lr=lr)   

# the problem is a min-max optimization: min_phi max_theta E_v[s(theta.x)] - E_0[s(theta.(z + phi))]

################################################# start training #################################################

# total number of epochs to train
num_epochs = 16
bs = 128 * 16  # batch size
n_bs = int(sample_size / bs)      # number of batches
Z = torch.randn(bs, dim)
Phi = np.zeros((num_epochs * n_bs // 4, dim))

for epoch in range(num_epochs):
    for _ in range(n_bs):        # for each batch
        d_opt.zero_grad()
        # train the discriminator
        x = X[(bs * _) : (bs * (_ + 1))].float()
        d = -dis(theta, x)
        d = d + dis(theta, gen(phi.float(), Z).detach())
        d.backward()
        d_opt.step()

        # train the generator
        g_opt.zero_grad()
        g = -dis(theta, gen(phi.float(), Z))
        g.backward()
        g_opt.step()
        if _ % 4 == 0:
            print("epoch: ", epoch, "batch: ", _, "phi: ", phi)
            Phi[(epoch * n_bs + _) // 4] = phi.detach().numpy()

np.save('adam_lr_' + str(lr) + '_r1_' + str(r1) + '_r2_' + str(r2), Phi)
