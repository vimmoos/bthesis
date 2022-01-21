import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/rewrite")
from interfaces import trainer, vanilla_trainer
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
import torch


@vanilla_trainer
def mse(model, data):
    x, y = data
    x_hat = model(x)
    loss = ((x - x_hat)**2).sum()
    loss.backward()
    return loss


@vanilla_trainer
def mse_kl(model, data):
    x, y = data
    x_hat = model(x)
    loss = ((x - x_hat)**2).sum() + model.encoder.kl
    loss.backward()
    return loss


def optimizers_AVB(model):
    disc_params = []
    ae_params = []
    for name, param in model.named_parameters():
        if 'disc' in name:
            disc_params.append(param)
        else:
            ae_params.append(param)

    return [torch.optim.Adam(disc_params), torch.optim.Adam(ae_params)]


@vanilla_trainer
def train_AVB(model, data):
    x, y = data
    log_d_prior, log_d_posterior, x_recon = model(x)
    disc_loss = torch.mean(
        binary_cross_entropy_with_logits(log_d_posterior,
                                         torch.ones_like(log_d_posterior)) +
        binary_cross_entropy_with_logits(log_d_prior,
                                         torch.zeros_like(log_d_prior)))

    recon_liklihood = -binary_cross_entropy(x_recon, x) * x.data.shape[0]

    ae_loss = torch.mean(log_d_posterior) - torch.mean(recon_liklihood)

    ae_loss.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    return (disc_loss, ae_loss)
