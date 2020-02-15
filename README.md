# ICLR-2020
code for our ICLR 2020 paper: Convergence Behaviour of Some Gradient-Based Methods on Bilinear Zero-Sum Games
## In the directory gmm:
* **gmm_"algorithm".py**: using the "algorithm" to learn Gaussian mixtures, Gauss-Seidel version
* **gmm_"algorithm"J.py**: using the "algorithm" to learn Gaussian mixtures, Jacobi version
* **model.py**: the neural net architecture
## In the directory shift:
* **shift_"algorithm".py**: using the "algorithm" to learn the mean of a Gaussian, Gauss-Seidel version
* **shift_"algorithm"J.py**: using the "algorithm" to learn the mean of a Gaussian, Jacobi version
* **model.py**: the discriminator and the generator
## Acronyms: 
* **gd/sgd**: gradient descent
* **m**: momentum / heavy ball
* **eg**: extra-gradient
* **ogd**: optimistic gradient descent
