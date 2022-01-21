import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/rewrite")
import torchvision
import utils as u
from networks import network
from vanilla import Decoder
import torch
import matplotlib.pyplot as plt


@network({
    'linear0': (torch.nn.Linear, 'input', 'hidden'),
    'mu': (torch.nn.Linear, 'hidden', 'latent'),
    'sigma': (torch.nn.Linear, 'hidden', 'latent')
})
def VariationalEncoder(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.relu(self.linear0(x))
    mu = self.mu(x)
    sigma = torch.exp(self.sigma(x))
    z = mu + sigma * u.normal.sample(mu.shape)
    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - (1 / 2)).sum()
    return z


@network({
    'encoder': (VariationalEncoder, 'hidden', 'input', 'latent'),
    'decoder': (Decoder, 'hidden', 'latent', 'input')
})
def VAE(self, x):
    return self.decoder(self.encoder(x))
