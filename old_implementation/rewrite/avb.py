import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/rewrite")
import utils as u
from networks import network
import torch
from torch.nn import Linear
from torch.autograd import Variable


def sample_prior(latent, s, training=True):
    return (Variable(
        torch.normal(torch.zeros(
            (s.data.shape[0], latent)), torch.ones((s.data.shape[0], latent))))
            if training else Variable(torch.zeros((s.data.shape[0], latent))))


@network({
    'layer0': (Linear, ['input', 'latent'], 'hidden'),
    'layer1': (Linear, 'hidden', 'latent'),
    'latent': (u.identity, 'latent')
})
def Encoder(self, x):
    """ test documentation"""
    i = torch.cat((x, sample_prior(self.latent, x, self.training)), dim=1)
    h = self.relu(self.layer0(i))
    return self.layer1(h)


@network((Linear, 'latent', 'hidden'), (Linear, 'hidden', 'output'))
def Decoder(self, z):
    i = self.relu(self.layer0(z))
    h = self.sigmoid(self.layer1(i))
    return h


@network((Linear, ['input', 'latent'], 'hidden'), (Linear, 'hidden', 1))
def Discriminator(self, x, z):
    i = torch.cat((x, z), dim=1)
    h = self.relu(self.layer0(i))
    return self.layer1(h)


@network({
    'encoder': (Encoder, 'latent', 'input', 'henc'),
    'decoder': (Decoder, 'latent', 'input', 'hdec'),
    'discriminator': (Discriminator, 'latent', 'input', 'hdisc'),
    'latent': (u.identity, 'latent')
})
def AVB(self, x):
    z_p = sample_prior(self.latent, x, self.training)

    z_q = self.encoder(x)
    log_d_prior = self.discriminator(x, z_p)
    log_d_posterior = self.discriminator(x, z_q)

    x_recon = self.decoder(z_q)

    return log_d_prior, log_d_posterior, x_recon
