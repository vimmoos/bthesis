import inspect as i
from typing import List

import torch
from torch import nn


class A(nn.Module):
    def __init__(self, x, y=1):
        super().__init__()
        self.lin = nn.Linear(x, y)

    def forward(self, x):
        return self.lin(x)


# def hparms(self) -> List[int]:
#     return [self.lin.in_features, self.lin.out_features]


# x = torch.jit.script(A())


# print(x.hparms())


def pippo(y, x=1):
    return x


@with_hparams
class B(nn.Module):
    def __init__(self, x: int, y=1, z=nn.Identity):
        super().__init__()
        self.lin = nn.Linear(x, y)

    def forward(self, x):
        return self.lin(x)
