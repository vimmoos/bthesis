import torch
from dataclasses import dataclass, field
from typing import List, Iterable, Callable, Dict


@dataclass(init=True, repr=True)
class DataLoader():
    _load: Callable

    def __call__(self, data):
        self.train, self.test = self._load(data)

    @property
    def test(self):
        if self.test == None:
            raise Exception("No data available")
        return self.test

    @property
    def train(self):
        if self.test == None:
            raise Exception("No data available")
        return self.test


dataloader = lambda fn: lambda: DataLoader(fn)


@dataclass(init=True, repr=True)
class Trainer():
    epochs: int

    data: DataLoader

    _step: Callable

    _epoch_step: Callable

    def __call__(self,
                 model: object,
                 optimizer: List[torch.optim.Optimizer] = None):
        self.optimizer = [torch.optim.Adam(model.parameters())
                          ] if optimizer == None else optimizer
        for epoch in range(self.epochs):
            for x in self.data.train:
                loss = self._step(self, model, x)
            print(f"loss after epoch {epoch} -> {loss}")
        return model


trainer = lambda fn: lambda data, epochs=20, epoch_step=lambda *args, **kwargs: None: Trainer(
    epochs, data, fn, epoch_step)


def vanilla_trainer(fn):
    @trainer
    def inner(self, model, data):
        for opt in self.optimizer:
            opt.zero_grad()
        loss = fn(model, data)
        for opt in self.optimizer:
            opt.step()
        return loss

    return inner


@dataclass(init=True, repr=True)
class Tester():
    data: DataLoader

    _test: Callable

    def __call__(self, model: object):
        return {x: self._test(model(x), x) for x in self.data.test}


tester = lambda fn: lambda data: Tester(data, fn)
