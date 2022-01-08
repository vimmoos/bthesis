"""TODO."""

import os

import torch

import thesis.runner as r
from thesis.ednconf import load_and_resolve
from thesis.logger import CsvLogger

os.chdir("/home/vimmoos/thesis-folder/thesis")


def testaruolo(conf="test"):
    args = load_and_resolve(conf)
    for k, v in args["data"].items():
        args[f"{k}_data"] = v
    del args["data"]
    if conf == "avb":
        args["model"] = torch.jit.script(args["model"])
    else:
        args["model"] = torch.jit.trace(
            args["model"], next(iter(args["train_data"]))[0], strict=False
        )
    r.run(**args)
    return args


def do_all():
    # ret = {}
    with CsvLogger(
        {
            "AVB": ["prefix", "loss_ae", "loss_disc"],
            "VAE": ["prefix", "loss_ae"],
            "VanillaAutoencoder": ["prefix", "loss_ae"],
        },
    ):
        ret = {
            "avb": testaruolo("avb"),
            "vae": testaruolo("vae"),
            "van": testaruolo("test"),
        }
    return ret


if __name__ == "__main__":
    for _ in range(10):
        do_all()
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
