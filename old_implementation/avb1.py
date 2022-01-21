import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class AVB(nn.Module):
    def __init__(self, input_size, latent, henc, hdec, hdisc):
        super().__init__()
        self.input = input_size
        self.latent = latent
        self.henc = henc
        self.hdec = hdec
        self.hdisc = hdisc

        self.dec_l1 = torch.nn.Linear(latent, hdec)
        self.dec_l2 = torch.nn.Linear(hdec, input_size)

        self.enc_l1 = torch.nn.Linear(input_size + latent, henc)
        self.enc_l2 = torch.nn.Linear(henc, latent)

        self.disc_l1 = torch.nn.Linear(input_size + latent, hdec)
        self.disc_l2 = torch.nn.Linear(hdec, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def sample_prior(self, s):
        if self.training:
            m = torch.zeros((s.data.shape[0], self.latent))
            std = torch.ones((s.data.shape[0], self.latent))
            d = Variable(torch.normal(m, std))
        else:
            d = Variable(torch.zeros((s.data.shape[0], self.latent)))

        return d

    def discriminator(self, x, z):
        i = torch.cat((x, z), dim=1)
        h = self.relu(self.disc_l1(i))
        return self.disc_l2(h)

    def sample_posterior(self, x):
        i = torch.cat((x, self.sample_prior(x)), dim=1)
        h = self.relu(self.enc_l1(i))
        return self.enc_l2(h)

    def decoder(self, z):
        i = self.relu(self.dec_l1(z))
        h = self.sigmoid(self.dec_l2(i))
        return h

    def forward(self, x):
        z_p = self.sample_prior(x)

        z_q = self.sample_posterior(x)
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


def plot_latent_AVB(model, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = model.sample_posterior(torch.reshape(torch.squeeze(x), (512, 784)))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def testaroulo_AVB():

    model = AVB(784, 2, 512, 512, 512)

    data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data', transform=torchvision.transforms.ToTensor(), download=True),
                                       batch_size=512,
                                       shuffle=True)
    m = train_AVB(model, data)

    plot_latent_AVB(m, data)
    plt.savefig("adversarial variational bayes")
    plt.clf()
    plt.cla()
    return m
