"""TODO."""


import thesis.losses as losses
import thesis.optimizers as opt
import thesis.utils as u


class Trainer:
    """TODO."""

    epochs: int = 20
    man_optims: opt.MOptims
    man_losses: losses.MLosses

    def __init__(
        self,
        epochs: int = 20,
        **kwargs,
    ):
        """Initialize class."""
        super().__init__()
        self.man_optims = opt.MOptims(**kwargs)
        self.man_losses = losses.MLosses(**kwargs)
        self.epochs = epochs

    def train(self, model, data, floss):
        """TODO."""
        self.man_optims.ignite(model)
        model.train()
        for ep in range(self.epochs):
            self.man_losses.reset_losses()
            for x, _ in data:
                self.man_optims.zero_grad()
                out = model(x)
                ls = floss(**u.sel_args(out, floss), x=x)
                self.man_losses.compute_backward(ls)
                self.man_optims.step()
            print(f"epoch {ep}")
            self.man_losses.print_losses()
        return model


# {
#     "ae": {
#         "optim": {
#             "name": optim.Adam,
#             "args": {"lr": 2},
#         },
#         "backward": {
#             "retain_graph": True,
#         },
#         # "scheduler": {
#         #     "name": optim.lr_scheduler.ExponentialLR,
#         #     "args": [],
#         # },
#     },
#     "disc": {
#         "optim": optim.Adam,
#         "backward": {
#             "retain_graph": True,
#         },
#         # "scheduler": {
#         #     "name": optim.lr_scheduler.ExponentialLR,
#         #     "args": [],
#         # },
#     },
# }
