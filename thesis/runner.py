"""TODO."""
from typing import Callable

import torch
from torch import nn

import thesis.losses as losses
import thesis.optimizers as opt
import thesis.utils as u
from thesis.logger import log


@torch.no_grad()
def test(
    mlosses: losses.MLosses,
    model: nn.Module,
    data,
    floss,
    floss_args,
):
    """TODO."""
    model.eval()
    with mlosses("TESTING ", data) as loss:
        for x, _ in data:
            out = model(x)
            ls = floss(**u.sel_args_l(out, floss_args), x=x)
            loss.add_losses(ls)


def train(
    mlosses: losses.MLosses,
    mopts: opt.MOptims,
    model: nn.Module,
    data,
    floss,
    floss_args,
):
    """TODO."""
    model.train()
    with mlosses("TRAINING ", data) as loss:
        for x, _ in data:
            with mopts:
                out = model(x)
                ls = floss(**u.sel_args_l(out, floss_args), x=x)
                loss.compute_backward(ls)
                loss.add_losses(ls)


def run(
    model: nn.Module,
    floss: Callable,
    train_data,
    test_data,
    epochs: int = 20,
    model_name: str = None,
    **kwargs,
):
    floss_args = u.get_args_name(floss)
    floss = torch.jit.script(floss)
    # mlosses = torch.jit.script(losses.MLosses(kwargs))
    model_name = model_name if model_name else getattr(model, "original_name", "test")
    mlosses = losses.MLosses(
        **kwargs,
        model_name=model_name,
    )

    # mopts = torch.jit.script(opt.MOptims(**kwargs))
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
    log(model_name)
    for ep in range(epochs):
        log(ep, end=" ", flush=True)
        mlosses.epoch = ep
        log("TRAINING", end=" ", flush=True)
        train(**train_arg)
        log("DONE", end=" ", flush=True)
        # dump train losses
        log("TESTING", end=" ", flush=True)
        test(**test_arg)
        log("DONE")
        # dump test losses
