"""Encoder module.

It store all the needed implementation for different type of
autoencoders

The current implementations are:

  + VariationalEncoder

  + AVBEncoder

"""
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn

import thesis.arch.utils as ua
import thesis.utils as u


class VariationalEncoder(nn.Module):
    """Create Variational Encoder.

    Takes a list of tuple where each tuple indicate a layer input-output
    dimensions, an intermediate activation function which will be
    interleaved between the layers and lastly a 'closing' activation
    function which will be called on both results of the mean and std
    layers

    At each step it returns a dict containing the following keys:
      + *mu*  -> the mean of the distribution
      + *var* -> the log variance of the distribution

    *NOTE*: The main peculiarity of a Variational Encoder is that
    instead of compressing information into a latent space, it
    compress the information in two layers. The first indicating the
    mean and the second the standard deviation of the latent
    distribution.
    """

    def __init__(
        self,
        layers: List[Tuple[int, int]],
        inter_act: Type[nn.Module] = nn.ReLU,
        final_act: Type[nn.Module] = nn.Identity,
    ):
        """Initialize the class."""
        super().__init__()
        self.seq = ua.make_linear_seq(layers[:-1], inter_act, inter_act)
        self.mu = nn.Linear(*layers[-1])
        self.logvar = nn.Linear(*layers[-1])
        self.final_act = final_act()

    def forward(self, x) -> Dict:
        """Make a single step of the VE.

        Returns a dict containing the following keys:
           + *mu*  -> the mean of the distribution
           + *var* -> the log variance of the distribution
        """
        h1 = self.seq(x)
        return {
            "mu": self.final_act(self.mu(h1)),
            "var": self.final_act(self.logvar(h1)),
        }


class AVBEncoder(nn.Module):
    """Adversarial Variational Bayes Encoder.

    Takes the same __init__ params as all the other encoders.
    It is almost equal to a Sequential Encoder, with the only
    difference that during each steps it also takes a prior
    distribution, and it concats it toghether with the input.

    At each step it returns a dict containing the following keys:
      + *out* -> the output of the step
    """

    def __init__(
        self,
        layers: List[Tuple[int, int]],
        latent: int,
        inter_act: Type[nn.Module] = nn.ReLU,
        final_act: Type[nn.Module] = nn.Identity,
    ):
        """Initialize the class."""
        super().__init__()
        layers[0] = (layers[0][0] + latent, layers[0][1])
        self.seq = u.apply(ua.make_linear_seq, locals())

    def forward(self, x, prior) -> Dict:
        """Make a single step of the AVBE.

        Returns a dict containing the following keys:
           + out -> the output of the step
        """
        print(x.shape)
        print(prior.shape)
        print(torch.cat((x, prior), dim=1).shape)
        return {"out": self.seq(torch.cat((x, prior), dim=1))}
