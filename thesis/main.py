"""TODO."""

import os
# import random
from collections.abc import Mapping

import matplotlib.pyplot as plt
# import numpy as np
import torch
from torch.distributions import Normal

import thesis.runner as r
from thesis.ednconf import load_and_resolve
from thesis.logger import CsvLogger, NoOpLogger
from thesis.losses import MLoss

# from pprint import pprint


os.chdir("/home/vimmoos/thesis-folder/thesis")


def testaruolo(conf="test"):
    args = load_and_resolve(conf)
    for k, v in args["data"].items():
        args[f"{k}_data"] = v
    del args["data"]

    args["mloss"] = MLoss(**args["loss"])
    del args["loss"]
    hparams = args["model"][1]

    args["model"] = args["model"][0]

    # TODO SEED STUFF
    # seed = args.get("seed", 1)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    if args.get("seed"):
        del args["seed"]
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # TODO change this!!!
    if "avb" in conf:
        args["model"] = torch.jit.script(args["model"])
    else:
        args["model"] = torch.jit.trace(
            args["model"], next(iter(args["train_data"]))[0], strict=False
        )
    model_name = args.get("name", conf)
    args["stats"] = r.Stats(
        hparams=hparams,
        model_name=model_name,
    )
    args["validate"] = [
        MLoss(**x, validation=True)
        if isinstance(x, Mapping)
        else MLoss(
            x,
            validation=True,
        )
        for x in args["validate"]
    ]

    with CsvLogger(
        {
            model_name: [
                "prefix",
                "loss_ae",
                "loss_disc",
                "loss_fun",
            ]
        }
    ):

        # with NoOpLogger():
        r.run(**args)
    return args


def do_all():
    print("START VAE")
    vae = testaruolo("vae")
    print("START AVB")
    avb = testaruolo("avb")
    print("START VANILLA")
    van = testaruolo("vanilla")
    return {
        "avb": avb,
        "vae": vae,
        "van": van,
    }


def plot_latent_vae(model, data, name="test", num_batches=100):
    for i, (x, y) in enumerate(data):
        latent = model.encoder(x)

        std = latent["var"].mul(0.5).exp_()
        z = Normal(latent["mu"], std).rsample()
        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig(name)
    # plt.show()
    plt.clf()
    plt.cla()


def plot_latent_vanilla(model, data, name="test", num_batches=100):
    model.eval()
    for i, (x, y) in enumerate(data):

        z = model.encoder(x)["out"]

        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig(name)
    # plt.show()
    plt.clf()
    plt.cla()


from thesis.arch.avb import sample_prior


def plot_latent(model, data, name="test", num_batches=100):
    model.eval()
    for i, (x, y) in enumerate(data):

        z_p = sample_prior(model.latent, x, model.training)
        z = model.encoder(x, z_p)["out"]

        z = z.detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            plt.colorbar()
            break
    plt.savefig(name)
    # plt.show()
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    # for _ in range(10):
    #     do_all()
    for _ in range(10):
        testaruolo("vae")
    # args = testaruolo("test")
    # testaruolo("avb")
    # do_all()
    # pr = cProfile.Profile()
    # pr.enable()
    # testaruolo("avb")
    # pr.disable()
    # result = io.StringIO()
    # pstats.Stats(pr, stream=result).print_stats()
    # result = result.getvalue()
    # # chop the string into a csv-like buffer
    # result = "ncalls" + result.split("ncalls")[-1]
    # result = "\n".join(
    #     [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
    # )
    # # save it to disk

    # with open("test.csv", "w+") as f:
    #     # f=open(result.rsplit('.')[0]+'.csv','w')
    #     f.write(result)
