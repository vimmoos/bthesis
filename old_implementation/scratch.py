import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/")
from model import network
import torch


@network(('Linear', 'input', 'hidden'), ('Linear', 'hidden', 'latent'))
class Encoder(torch.nn.Module):
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.layer0(x))
        return self.layer1(x)


@network(('Linear', 'latent', 'hidden'), ('Linear', 'hidden', 'output'))
class Decoder(torch.nn.Module):
    def forward(self, z):
        z = self.relu(self.layer0(z))
        z = torch.sigmoid(self.layer1(z))
        return z.reshape((-1, 1, 28, 28))


@network({
    'encoder': (Encoder, 'input', 'hidden', 'latent'),
    'decoder': (Decoder, 'latent', 'hidden', 'input')
})
class Autoencoder(nn.Module):
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(autoencoder: Autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print(f"epoch {epoch}")
    return autoencoder


@network({
    'linear0': (torch.nn.Linear, 'input', 'hidden'),
    'mu': (torch.nn.Linear, 'hidden', 'latent'),
    'sigma': (torch.nn.Linear, 'hidden', 'latent')
})
class VariationalEncoder(nn.Module):
    def __init__(self):
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.linear0(x))
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - (1 / 2)).sum()
        return z


@network({
    'encoder': (VariationalEncoder, 'input', 'hidden', 'latent'),
    'decoder': (Decoder, 'latent', 'hidden', 'input')
})
class VAE(nn.Module):
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_VAE(autoencoder: VAE, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
        print(f"epoch {epoch}")
    return autoencoder
