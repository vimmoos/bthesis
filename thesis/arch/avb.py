"""Watch more carefully the paper! https://arxiv.org/pdf/1701.04722.pdf."""

from typing import Dict, List, Tuple, Type

import torch
from torch import nn

import thesis.arch.encoders as enc
import thesis.arch.utils as ua
import thesis.utils as u


def sample_prior(latent_size: int, x: torch.Tensor, training: bool) -> torch.Tensor:
    """TODO."""
    if training:
        m = torch.zeros((x.data.shape[0], latent_size))
        std = torch.ones((x.data.shape[0], latent_size))
        return torch.normal(m, std).requires_grad_(True)
    return torch.zeros(x.data.shape[0], latent_size).requires_grad_(True)


class Discriminator(nn.Module):
    """TODO."""

    def __init__(
        self,
        layers: List[Tuple[int, int]],
        latent: int,
        inter_act: Type[nn.Module] = nn.ReLU,
        final_act: Type[nn.Module] = nn.Identity,
    ):
        """TODO."""
        super().__init__()
        layers[0] = (layers[0][0] + latent, layers[0][1])
        self.seq = u.apply(ua.make_linear_seq, locals())

    def forward(self, x, z) -> Dict[str, torch.Tensor]:
        """TODO."""
        return {"out": self.seq(torch.cat((x, z), dim=1))}


class AVB(nn.Module):
    """TODO."""

    def __init__(
        self,
        enc_layers: List[Tuple[int, int]],
        dec_layers: List[Tuple[int, int]],
        disc_layers: List[Tuple[int, int]],
        enc_inter_act: Type[nn.Module] = nn.ReLU,
        enc_final_act: Type[nn.Module] = nn.Identity,
        dec_inter_act: Type[nn.Module] = nn.ReLU,
        dec_final_act: Type[nn.Module] = nn.Sigmoid,
        disc_inter_act: Type[nn.Module] = nn.ReLU,
        disc_final_act: Type[nn.Module] = nn.Identity,
    ):
        """TODO."""
        kwargs = locals()
        super().__init__()
        self.latent = enc_layers[-1][1]
        self.encoder = torch.jit.script(
            enc.AVBEncoder(**u.sel_and_rm(kwargs, "enc_"), latent=self.latent)
        )
        self.decoder = torch.jit.script(u.sapply(ua.Sequential, kwargs, "dec_"))
        self.disc = torch.jit.script(
            Discriminator(**u.sel_and_rm(kwargs, "disc_"), latent=self.latent)
        )

    @torch.jit.ignore
    def parameters(self):
        """TODO."""
        disc_params, ae_params = [], []
        for name, param in self.named_parameters():
            if "disc" in name:
                disc_params.append(param)
            else:
                ae_params.append(param)

        return {"ae": ae_params, "disc": disc_params}

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """TODO."""
        z_p = sample_prior(self.latent, x, self.training)
        z_q = self.encoder(x, z_p)["out"]
        log_d_prior = self.disc(x, z_p)["out"]
        log_d_posterior = self.disc(x, z_q)["out"]
        recon_x = self.decoder(z_q)["out"]
        return {
            "out": recon_x,
            "logprior": log_d_prior,
            "logpost": log_d_posterior,
            "latent": z_q,
        }
