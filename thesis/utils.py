"""Utils module, contains misc utility functions."""

import inspect as i
import statistics as stats
from typing import Any, Callable, Dict

import torch
from torchvision import datasets, transforms


def sel_and_rm(kw: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Select only the entries which starts with the prefix and remove the prefix."""
    lpre = len(prefix)
    return {k[lpre:]: v for k, v in kw.items() if prefix == k[0:lpre]}


def sel_args(kw: Dict[str, Any], fun: Callable) -> Dict[str, Any]:
    """Select only the args needed by the function from the kw."""
    return {k: v for k, v in kw.items() if k in list(i.signature(fun).parameters)}


def sel_args_l(kw: Dict[str, Any], llargs: list) -> Dict[str, Any]:
    """Select only the args needed by the function from the kw."""
    return {k: v for k, v in kw.items() if k in llargs}


def get_args_name(fun: Callable):
    return list(i.signature(fun).parameters)


def apply(fun: Callable, kw: Dict[str, Any]):
    """TODO."""
    return fun(**sel_args(kw, fun))


def sapply(fun: Callable, kw: Dict[str, Any], prefix: str):
    """TODO."""
    return fun(**sel_and_rm(kw, prefix=prefix))


def load_data():
    """TODO."""
    tensor_transform = transforms.ToTensor()
    trans = transforms.Lambda(
        lambda x: tensor_transform(x).reshape(-1, 28 * 28).squeeze()
    )

    train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=trans,
    )
    test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=trans,
    )

    return {
        "train": torch.utils.data.DataLoader(
            dataset=train,
            batch_size=32,
            shuffle=True,
        ),
        "test": torch.utils.data.DataLoader(
            dataset=test,
            batch_size=32,
            shuffle=True,
        ),
    }


def msd(arr):
    """TODO."""
    return (stats.mean(arr), stats.stdev(arr))
