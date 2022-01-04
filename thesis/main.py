"""TODO."""

import os

os.chdir("/home/vimmoos/thesis-folder/thesis")


import cProfile
import io
import math
import pstats

import torch

import thesis.ednconf.core as c
import thesis.runner as r


def testaruolo(conf="test"):
    args = c.load_and_resolve(conf)
    for k, v in args["data"].items():
        args[f"{k}_data"] = v
    del args["data"]
    args["model"] = torch.jit.script(args["model"])
    r.run(**args)
    return args


def do_all():
    return {
        "avb": testaruolo("avb"),
        "vae": testaruolo("vae"),
        "van": testaruolo("test"),
    }


if __name__ == "__main__":
    testaruolo("test")
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
