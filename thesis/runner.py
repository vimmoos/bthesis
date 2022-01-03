"""TODO."""
from typing import Callable

import torch
from torch import nn

import thesis.optimizers as opt
import thesis.utils as u
from thesis import losses


@torch.inference_mode()
def test(mlosses: losses.MLosses, model: nn.Module, data, floss):
    """TODO."""
    model.eval()
    with mlosses("TESTING ", len(data.dataset)) as loss:
        for x, _ in data:
            out = model(x)
            ls = floss(**u.sel_args(out, floss), x=x)
            loss.add_losses(ls)


# @torch.enable_grad()
def train(mlosses: losses.MLosses, mopts: opt.MOptims, model: nn.Module, data, floss):
    """TODO."""
    model.train()
    with mlosses("TRAINING ", len(data.dataset)) as loss:
        for x, _ in data:
            with mopts:
                out = model(x)
                ls = floss(**u.sel_args(out, floss), x=x)
                loss.compute_backward(ls)
                loss.add_losses(ls)


def run(
    model: nn.Module,
    floss: Callable,
    train_data,
    test_data,
    epochs: int = 20,
    **kwargs
):
    mlosses = losses.MLosses(**kwargs)
    mopts = opt.MOptims(**kwargs)
    mopts.ignite(model)
    train_arg = {
        **u.sel_args(locals(), train),
        **u.sel_and_rm(locals(), "train_"),
    }
    test_arg = {
        **u.sel_args(locals(), test),
        **u.sel_and_rm(locals(), "test_"),
    }
    for ep in range(epochs):
        train(**train_arg)
        # dump train losses
        test(**test_arg)
        # dump test losses
