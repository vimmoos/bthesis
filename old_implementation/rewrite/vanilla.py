import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/rewrite")
from networks import network
from torch.nn import Linear
import torch


@network((Linear, 'input', 'hidden'), (Linear, 'hidden', 'latent'))
def Encoder(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.relu(self.layer0(x))
    return self.layer1(x)


@network((Linear, 'latent', 'hidden'), (Linear, 'hidden', 'output'))
def Decoder(self, z):
    z = self.relu(self.layer0(z))
    z = torch.sigmoid(self.layer1(z))
    return z.reshape((-1, 1, 28, 28))


@network({
    'encoder': (Encoder, 'hidden', 'input', 'latent'),
    'decoder': (Decoder, 'hidden', 'latent', 'input')
})
def Autoencoder(self, x):
    return self.decoder(self.encoder(x))
