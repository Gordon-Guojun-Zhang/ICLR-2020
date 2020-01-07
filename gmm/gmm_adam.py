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
import matplotlib.pyplot as plt
import matplotlib
import os


##################################### cuda/GPU ########################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# fixing the seed makes the convergence bad, no idea why
seed = 12
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


################################################# data preprocessing #############################################

class Dataset(data.TensorDataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, z):
      'Initialization, (x, y) is a 2d point array'
      self.pt = torch.from_numpy(z)  # stack along the column direction

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.pt)

  def __getitem__(self, index):
      'Generates one sample of data'
      # Load data and get label
      pt = self.pt[index]
      return pt



# np.linspace(start, stop, num)
# return a 1d grid
num_mix = 8
ths = np.linspace(0, 2 * np.pi * (num_mix - 1)/num_mix, num_mix)
xs, ys = 2 * np.cos(ths), 2 * np.sin(ths)

samples = 100000    # number of samples
K = np.random.randint(num_mix, size=samples)    # which mixture we choose
X = np.zeros(samples)
Y = np.zeros(samples)


for _ in range(samples):
    cx, cy = xs[K[_]], ys[K[_]]
    X[_], Y[_] = cx + np.random.randn() / 10, cy + np.random.randn() / 10

Z = np.stack((X, Y), axis=-1)

GMM = Dataset(Z)

###################################### visualize ###############################################################

#matplotlib.use('Agg')

# Load data
train_data = GMM

# create loader with data to iterate, batch size is 100
data_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
# number of batches, 100000 / 100 = 1000
num_batches = len(data_loader)


################################################# defining G and D #############################################
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

dis = DNet().float().to(device)   # generate a funtion


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

gen = GNet().float().to(device)    # generate a function

# hidden: torch.randn(size, 100)





################################################# optimizers #############################################
lr = 0.0002
d_opt = optim.Adam(dis.parameters(), lr=lr)   # use Adam, the first argument is the parameters to update
g_opt = optim.Adam(gen.parameters(), lr=lr)   # use Adam, the first argument is the parameters to update



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

# min -log(D(G(z)))
def train_gen(optimizer, fake_data):
    samples = fake_data.size(0)
    # reset the gradient
    optimizer.zero_grad()

    # sample the hidden distribution and generate the fake data
    prediction = dis(fake_data).to(device)
    # forward and backward
    error = criterion(prediction, torch.ones(samples, 1).to(device))
    error.backward()
    # update the weights
    optimizer.step()

    return error


# some samples from the hidden distribution
num_test_samples = 2000
hidden = torch.randn(num_test_samples, 100).to(device)


################################################# start training #################################################

logger = Logger(model_name='VGAN', data_name='GMM')  # VGAN stands for vanilla GAN

# total number of epochs to train
num_epochs = 200
Phi = np.zeros((num_epochs, num_test_samples, 2))
if not os.path.isdir('./gmm_adam'):
    os.mkdir('./gmm_adam')

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
        if (n_batch) % 1000 == 0:
            test_pt = gen(hidden).to(torch.device("cpu"))   # from GPU to CPU
            test_pt = test_pt.detach().numpy()
            Phi[epoch] = test_pt
            plt.clf()     # clear the plot every time to avoid stacking the cache
            sns.kdeplot(test_pt[:, 0], test_pt[:, 1], shade='True')
            plt.savefig('./gmm_adam/epoch_' + str(epoch) + '_batch_' + str(n_batch) + '.pdf')   # save image
            # Display status Logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, \
                    d_pred_fake)


np.save('gmm_adam_lr_' + str(lr), Phi)
