import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_size, hidden, latent):
        self.layer1 = torch.nn.Linear(input_size + latent, hidden)
        self.layer2 = torch.nn.Linear(hidden, latent)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        i = torch.cat((x, self.sample_prior(x)), dim=1)
        h = self.relu(self.enc_l1(i))
        return self.enc_l2(h)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden, latent):
        self.layer1 = torch.nn.Linear(latent, hidden)
        self.layer2 = torch.nn.Linear(hidden, input_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        i = self.relu(self.dec_l1(z))
        h = self.sigmoid(self.dec_l2(i))
        return h


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden, latent):
        self.disc_l1 = torch.nn.Linear(input_size + latent, hidden)
        self.disc_l2 = torch.nn.Linear(hidden, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, z):
        i = torch.cat((x, z), dim=1)
        h = self.relu(self.disc_l1(i))
        return self.disc_l2(h)


def sample_prior(latent, s, training=True):
    return (Variable(
        torch.normal(torch.zeros(
            (s.data.shape[0], latent)), torch.ones((s.data.shape[0], latent))))
            if training else Variable(torch.zeros((s.data.shape[0], latent))))


class AVB(nn.Module):
    def __init__(self, input_size, latent, henc, hdec, hdisc):
        super().__init__()
        self.latent = latent
        self.encoder = Encoder(input_size, henc, latent)
        self.decoder = Decoder(input_size, hdec, latent)
        self.discriminator = Discriminator(input_size, hdisc, latent)

    def forward(self, x):
        z_p = sample_prior(self.latent, x)

        z_q = self.encoder(x)
        log_d_prior = self.discriminator(x, z_p)
        log_d_posterior = self.discriminator(x, z_q)
        disc_loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_posterior, torch.ones_like(log_d_posterior)) +
            torch.nn.functional.binary_cross_entropy_with_logits(
                log_d_prior, torch.zeros_like(log_d_prior)))

        x_recon = self.decoder(z_q)
        recon_liklihood = -torch.nn.functional.binary_cross_entropy(
            x_recon, x) * x.data.shape[0]

        gen_loss = torch.mean(log_d_posterior) - torch.mean(recon_liklihood)

        return disc_loss, gen_loss


def train_AVB(model, data, epochs=20):
    disc_params = []
    ae_params = []
    for name, param in model.named_parameters():
        if 'disc' in name:
            disc_params.append(param)
        else:
            ae_params.append(param)

    disc_optimizer = torch.optim.Adam(disc_params)
    ae_optimizer = torch.optim.Adam(ae_params)

    for epoch in range(epochs):
        for x, y in data:
            if (x.shape[0] != 512): continue
            ae_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            disc_loss, ae_loss = model(
                torch.reshape(torch.squeeze(x), (512, 784)))
            ae_loss.backward(retain_graph=True)
            disc_loss.backward(retain_graph=True)
            ae_optimizer.step()
            disc_optimizer.step()
        print(f"done epoch { epoch}")
    return model
