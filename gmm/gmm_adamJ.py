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
from oadam import OAdam
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib
import setup, gmm_data, model

##################################### cuda/GPU ########################################################
device = setup.init_seed()
train_data = gmm_data.get_data()

# create loader with data to iterate, batch size is 100
data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
# number of batches, 100000 / 100 = 1000
num_batches = len(data_loader)

dis = model.DNet().float().to(device)   # generate a funtion
gen = model.GNet().float().to(device)    # generate a function

################################################# optimizers #############################################
lr = 0.0002
d_opt = OAdam(dis.parameters(), lr=0.0001, beta=0.0)   # use Adam, the first argument is the parameters to update
g_opt = OAdam(gen.parameters(), lr=0.0001, alpha=-2.0, beta=0.0)   # use Adam, the first argument is the parameters to update, simultaneous update


################################################# loss function #############################################
criterion = nn.BCELoss().to(device)  # binary cross entropy loss, nothing fancy but a dirty way to compute the log


################################################# wrap the training as functions ################################

# the problem is a min-max optimization: min_G max_D log(D(x)) + log(1 - D(G(z)))


# minimize -log(D(x)) - log (1 - D(G(z))), take one gradient descent step
# this is equivalent to max_D log(D(x)) + log(1 - D(G(z)))
def train(d_optimizer, g_optimizer, real_data, fake_data):
    samples = real_data.size(0)   # the number of samples
    # reset the gradient
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

    # train on real data
    prediction_real = dis(real_data).to(device)
    error_real = criterion(prediction_real, torch.ones(samples, 1).to(device)).to(device)  # -log(D(x))
    error_real.backward()       # autograd

    # train on fake data
    prediction_fake = dis(fake_data).to(device)
    error_fake = criterion(prediction_fake, torch.zeros(samples, 1).to(device)).to(device)  # -log(1 - D(G(z)))
    error_fake.backward()       # autograd

    # update the weights with the gradient
    d_optimizer.step()
    #for params in gen.parameters():
    #    print("g_grad: ", params.grad)
    g_optimizer.step()

    return error_real + error_fake, error_fake, prediction_real, prediction_fake


# some samples from the hidden distribution
num_test_samples = 2000
hidden = torch.randn(num_test_samples, 100).to(device)


################################################# start training #################################################

logger = Logger(model_name='VGAN', data_name='GMM_jacobi_adam')  # VGAN stands for vanilla GAN

# total number of epochs to train
num_epochs = 200
Phi = np.zeros((num_epochs, num_test_samples, 2))
if not os.path.isdir('./gmm_adamJ'):
    os.mkdir('./gmm_adamJ')

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
        fake_data = gen(torch.randn(samples, 100).to(device)).to(device)
        # we detach from the generator so that the gradient w.r.t. the generator is not computed

        # one-step training, compute the gradient and update
        d_error, g_error, d_pred_real, d_pred_fake = train(d_opt, g_opt, real_data, fake_data)
    
        # log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # display progress every few batches
        if (n_batch) % 1000 == 0:
            test_pt = gen(hidden).to(torch.device("cpu"))   # from GPU to CPU
            test_pt = test_pt.detach().numpy()
            Phi[epoch] = test_pt.copy()
            plt.clf()     # clear the plot every time to avoid stacking the cache
            sns.kdeplot(test_pt[:, 0], test_pt[:, 1], shade='True')
            plt.savefig('./gmm_adamJ/epoch_' + str(epoch) + '_batch_' + str(n_batch) + '.pdf')   # save image
            # Display status Logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, \
                    d_pred_fake)

np.save('gmm_adamJ_lr_' + str(lr), Phi)
