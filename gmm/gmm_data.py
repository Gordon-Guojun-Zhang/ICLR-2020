from torch.utils import data
import numpy as np
import torch

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

def get_data(num_mix=8, samples=100000):
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
    return GMM

