"""TODO."""
from typing import List

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from thesis.logger import log
from thesis.losses.manager import MLoss
from thesis.runner.stats import Stats


@torch.no_grad()
def test(
    mloss: MLoss,
    model: nn.Module,
    stats: Stats,
    test_data: DataLoader,
):
    """TODO."""
    model.eval()
    with stats("TESTING", test_data, mloss) as add:
        for x, _ in test_data:
            out = model(x)
            add(mloss(target=x, outs=out))


def train(
    mloss: MLoss,
    model: nn.Module,
    stats: Stats,
    train_data: DataLoader,
):
    """TODO."""
    model.train()
    with stats("TRAINING", train_data, mloss) as add:
        for x, _ in train_data:
            with mloss.moptims:
                out = model(x)
                add(mloss(target=x, outs=out))
                mloss.compute_backward()


def validation(
    mloss: MLoss,
    model: nn.Module,
    stats: Stats,
    test_data: DataLoader,
):
    with stats("VALIDATE", test_data, mloss) as add:
        for x, _ in test_data:
            out = model(x)
            add(mloss(target=x, outs=out))


def run(
    model: nn.Module,
    mloss: MLoss,
    stats: Stats,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    validate: List[MLoss],
):

    mloss.ignite(model)
    log(stats.model_name)
    for ep in range(epochs):
        log(ep, end=" ", flush=True)
        # mlosses.epoch = ep
        log("TRAINING", end=" ", flush=True)
        train(
            mloss=mloss,
            model=model,
            train_data=train_data,
            stats=stats,
        )
        log("DONE TESTING", end=" ", flush=True)
        test(
            mloss=mloss,
            model=model,
            test_data=test_data,
            stats=stats,
        )
        log("DONE")
    for loss in validate:
        log("VALIDATE", end=" ", flush=True)
        validation(
            mloss=loss,
            model=model,
            test_data=test_data,
            stats=stats,
        )
