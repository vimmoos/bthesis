import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

representation_size = 2
input_size = 4
n_samples = 2000
batch_size = 500
gen_hidden_size = 200
enc_hidden_size = 200
disc_hidden_size = 200

n_samples_per_batch = n_samples // input_size

y = np.array(
    [i for i in range(input_size) for _ in range(n_samples_per_batch)])

d = np.identity(input_size)
x = np.array([d[i] for i in y], dtype=np.float32)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.gen_l1 = torch.nn.Linear(representation_size, gen_hidden_size)
        self.gen_l2 = torch.nn.Linear(gen_hidden_size, input_size)

        self.enc_l1 = torch.nn.Linear(input_size + representation_size,
                                      enc_hidden_size)
        self.enc_l2 = torch.nn.Linear(enc_hidden_size, representation_size)

        self.disc_l1 = torch.nn.Linear(input_size + representation_size,
                                       disc_hidden_size)
        self.disc_l2 = torch.nn.Linear(disc_hidden_size, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def sample_prior(self, s):
        if self.training:
            m = torch.zeros((s.data.shape[0], representation_size))
            std = torch.ones((s.data.shape[0], representation_size))
            d = Variable(torch.normal(m, std))
        else:
            d = Variable(torch.zeros((s.data.shape[0], representation_size)))

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
        i = self.relu(self.gen_l1(z))
        h = self.sigmoid(self.gen_l2(i))
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


model = VAE()
disc_params = []
gen_params = []
for name, param in model.named_parameters():

    if 'disc' in name:

        disc_params.append(param)
    else:
        gen_params.append(param)

disc_optimizer = torch.optim.Adam(disc_params, lr=1e-3)
gen_optimizer = torch.optim.Adam(gen_params, lr=1e-3)


def train(epoch, batches_per_epoch=501, log_interval=500):
    model.train()

    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)])
        data = Variable(data, requires_grad=False)

        discrim_loss, gen_loss = model(data)

        gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)

        disc_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)

        disc_optimizer.step()
        gen_optimizer.step()

        if (i % log_interval == 0) and (epoch % 1 == 0):
            #Print progress
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * batches_per_epoch,
                discrim_loss / len(data), gen_loss / len(data)))

    print('====> Epoch: {} done!'.format(epoch))


torch.autograd.set_detect_anomaly(True)

for epoch in range(1, 15):
    train(epoch)
