"""TODO."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def ELBOLoss(
    out: Tensor, mu: Tensor, var: Tensor, x: Tensor
) -> Dict[str, torch.Tensor]:
    """Compute the ELBO loss.

    *NOTE* it uses the binary cross entropy to compare out to x

    Returns a map containing the following keys:
       + out -> the ELBO loss
    """
    KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    # normalize by Batchsize * inputsize
    KLD /= x.shape[0] * x.shape[1]
    return {"ae": F.binary_cross_entropy(out, x) + KLD}


__mseloss = torch.nn.MSELoss()


def mse(out, x):
    return {"ae": __mseloss(out, x)}


def ELBOWithDiscLoss(
    out: Tensor, logprior: Tensor, logpost: Tensor, x: Tensor
) -> Dict[str, torch.Tensor]:
    """Compute the ELBO loss and the discriminator loss.

    *NOTE* it uses the binary cross entropy for both the *disc* and *ae* losses

    Returns a map containing the following keys:
      + disc -> the discriminator loss
      + ae   -> the autoencoder loss
    """
    disc_loss = torch.mean(
        F.binary_cross_entropy_with_logits(logprior, torch.zeros_like(logprior))
        + F.binary_cross_entropy_with_logits(logpost, torch.ones_like(logpost))
    )
    recon_lh = -F.binary_cross_entropy(out, x) * x.data.shape[0]
    return {"disc": disc_loss, "ae": torch.mean(logpost) - torch.mean(recon_lh)}
