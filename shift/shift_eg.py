import torch
from torch import optim
from torch import tensor
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

################################################# parameters #############################################
alpha = 0.02
beta = 0.004
gamma = beta / alpha
print("alpha: ", alpha, "gamma: ", gamma)

# the problem is a min-max optimization: min_phi max_theta E_v[s(theta.x)] - E_0[s(theta.(z + phi))]

################################################# start training #################################################

# total number of epochs to train
num_epochs = 32
print("number of epochs: ", num_epochs)
bs = 128 * 16   # batch size
n_bs = 128      # number of batches
Z = torch.randn(bs, dim)
Phi = np.zeros((num_epochs * n_bs // 4, dim))

for epoch in range(num_epochs):
    for _ in range(n_bs):        # for each batch
        # train the discriminator
        if theta.grad is not None:
            theta.grad.zero_()
        if phi.grad is not None:
            print("phi grad nonzero")
            phi.grad.zero_()
        x = X[(bs * _) : (bs * (_ + 1))].float()
        
        d = dis(theta, x) - dis(theta, gen(phi.float(), Z))
        d.backward()
        # compute x^{t+1/2}, y^{t+1/2}
        theta_half = theta.detach()
        theta_half = theta.data + gamma * theta.grad
        theta_half.requires_grad_()
        phi_half = phi.detach()
        phi_half = phi.data - gamma * phi.grad
        phi_half.requires_grad_()


        # x^{t+1} = x^t + alpha * nabla_x f(x^{t+1/2}, y^{t+1/2})
        d = dis(theta_half, x) - dis(theta_half, gen(phi_half.float().detach(), Z))
        d.backward()
        theta.data = theta.data + alpha * theta_half.grad
        theta.requires_grad_()
        
        
        # compute x^{t+3/2}, y^{t+3/2}
        d = dis(theta, x) - dis(theta, gen(phi.float(), Z))
        d.backward()
        theta_half = theta.detach()
        theta_half = theta.data + gamma * theta.grad
        theta_half.requires_grad_()
        phi_half = phi.detach()
        phi_half = phi.data - gamma * phi.grad
        phi_half.requires_grad_()

        # x^{t+1} = x^t + alpha * nabla_x f(x^{t+1/2}, y^{t+1/2})
        g = - dis(theta_half, gen(phi_half.float(), Z))
        g.backward()
        phi = phi.data - alpha * phi_half.grad
        phi.requires_grad_()

        if _ % 4 == 0:
            print("epoch: ", epoch, "batch: ", _, "phi: ", phi)
            Phi[(epoch * n_bs + _) // 4] = phi.detach().numpy()

np.save("eg_gamma_" + str(gamma) + "_alpha_" + str(alpha), Phi)
