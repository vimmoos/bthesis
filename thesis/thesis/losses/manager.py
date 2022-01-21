"""TODO."""
import math
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn

import thesis.utils as u
from thesis.losses.optimizers import MOptims


class MLoss:
    """TODO."""

    floss: nn.Module
    floss_args: List[str]
    losses_arg: Dict[str, Dict[str, Any]]
    losses_ord: Optional[List[str]]
    moptims: MOptims
    current_loss: torch.Tensor
    name: str

    def __init__(
        self,
        name: Type[nn.Module],
        args: Dict[str, Any] = {},
        validation: bool = False,
        **kwargs
    ):
        """Initialize class."""
        self.name = name.__name__
        self.floss_args = [x for x in u.get_args_name(name.forward) if x != "self"]
        self.floss = torch.jit.script(name(**args))
        if not validation:
            self.moptims = MOptims(**kwargs)
            self.losses_arg = {k: v.get("backward", {}) for k, v in kwargs.items()}
            self.losses_ord = [
                x[0]
                for x in sorted(
                    kwargs.items(), key=lambda x: x[1].get("order", math.inf)
                )
            ]

    def ignite(self, model: nn.Module):
        """Given a model initialize the optimizers."""
        self.moptims.ignite(model)

    def __call__(self, target, outs):
        """TODO."""
        self.current_loss = self.floss(
            **u.sel_args_l(outs, self.floss_args), target=target
        )
        return self.current_loss

    # maybe vectorize it
    def compute_backward(self):
        """Compute all the gradients using .backward and adds the losses values."""
        for k in self.losses_ord:
            self.current_loss[k].backward(**self.losses_arg[k])
