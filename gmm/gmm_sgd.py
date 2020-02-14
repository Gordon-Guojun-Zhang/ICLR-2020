'''https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f'''
import torch
from torch import nn, optim
# this seems to be discouraged
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
# for visualization
from utils import Logger
from torch.utils import data
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from setup import *
from gmm_data import *
from model import *


device = init_seed()
train_data = get_data()

# create loader with data to iterate, batch size is 100
data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
# number of batches, 100000 / 100 = 1000
num_batches = len(data_loader)

dis = DNet().float().to(device)   # generate a funtion

gen = GNet().float().to(device)    # generate a function


################################################# optimizers #############################################
lr = 0.01
d_opt = optim.SGD(dis.parameters(), lr=0.01)   # use Adam, the first argument is the parameters to update
g_opt = optim.SGD(gen.parameters(), lr=0.01)   # use Adam, the first argument is the parameters to update



################################################# loss function #############################################
criterion = nn.BCELoss().to(device)  # binary cross entropy loss, nothing fancy but a dirty way to compute the log


################################################# wrap the training as functions ################################

# the problem is a min-max optimization: min_G max_D log(D(x)) + log(1 - D(G(z)))


# minimize -log(D(x)) - log (1 - D(G(z))), take one gradient descent step
# this is equivalent to max_D log(D(x)) + log(1 - D(G(z)))
def train_dis(optimizer, real_data, fake_data):
    samples = real_data.size(0)   # the number of samples
    # reset the gradient
    optimizer.zero_grad()

    # train on real data
    prediction_real = dis(real_data).to(device)
    error_real = criterion(prediction_real, torch.ones(samples, 1).to(device)).to(device)  # -log(D(x))
    error_real.backward()       # autograd

    # train on fake data
    prediction_fake = dis(fake_data).to(device)
    error_fake = criterion(prediction_fake, torch.zeros(samples, 1).to(device)).to(device)  # -log(1 - D(G(z)))
    error_fake.backward()       # autograd

    # update the weights with the gradient
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake


def train_gen(optimizer, fake_data):
    samples = fake_data.size(0)
    # reset the gradient
    optimizer.zero_grad()

    # sample the hidden distribution and generate the fake data
    prediction = dis(fake_data).to(device)
    # forward and backward
    error = -criterion(prediction, torch.zeros(samples, 1).to(device))
    error.backward()
    # update the weights
    optimizer.step()

    return error


# some samples from the hidden distribution
num_test_samples = 2000
hidden = torch.randn(num_test_samples, 100).to(device)


################################################# start training #################################################

logger = Logger(model_name='VGAN', data_name='GMM_sgd')  # VGAN stands for vanilla GAN

# total number of epochs to train
num_epochs = 50
Phi = np.zeros((num_epochs * 10, num_test_samples, 2))
if not os.path.isdir('./gmm_sgd'):
    os.mkdir('./gmm_sgd')


for epoch in range(num_epochs):
    for n_batch, zb in enumerate(data_loader):
        # the use of enumerate: enumerate(list, start_number=0)
        # usually enumerate is used together with for
        # for number, sth in enumerate(list):
        #      number: 0, sth: the element
        # in such a case, n_batch starts from 0
        # real_batch is the images in this batch
        # _ is the labels in this batch. We don't need them
        
        samples = zb.size(0)   # the batch size
        

        # train the discriminator first
        real_data = zb.float().to(device)
        fake_data = gen(torch.randn(samples, 100).to(device)).detach().to(device)
        # we detach from the generator so that the gradient w.r.t. the generator is not computed

        # one-step training, compute the gradient and update
        d_error, d_pred_real, d_pred_fake = train_dis(d_opt, real_data, fake_data)
    
        # train the generator
        # generate the fake data
        
        fake_data = gen(torch.randn(samples, 100).to(device))
        g_error = train_gen(g_opt, fake_data)
        
        # log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # display progress every few batches
        if (n_batch) % 100 == 0:
            test_pt = gen(hidden).to(torch.device("cpu"))   # from GPU to CPU
            test_pt = test_pt.detach().numpy()
            Phi[epoch * 10 + n_batch // 100] = test_pt.copy()
            plt.clf()     # clear the plot every time to avoid stacking the cache
            sns.kdeplot(test_pt[:, 0], test_pt[:, 1], shade='True')
            plt.savefig('./gmm_sgd/epoch_' + str(epoch) + '_batch_' + str(n_batch) + '.pdf')   # save image
            # Display status Logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, \
                    d_pred_fake)

np.save("gmm_sgd_lr_" + str(lr), Phi)
