import numpy as np

dim = 2   # dimension of the Gaussian distribution
sample_size = 128 * 128 * 16  # number of samples
X = np.zeros((sample_size, dim))
cx = np.ones(dim)   # center

for _ in range(sample_size):
    X[_] = cx + np.random.randn(dim)

np.save('one_gaussian', X)
