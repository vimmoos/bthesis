import statistics as stats
from pprint import pprint

import torch
import torch.optim as optim
from torch.nn import Linear, ReLU, Sequential, Sigmoid

import thesis.ednconf.core as c
from thesis.arch.avb import AVB
from thesis.losses import ELBOWithDiscLoss


def msd(arr):
    return (stats.mean(arr), stats.stdev(arr))


def train(model, data, epochs=20):
    model.train()
    ps = model.parameters()
    ae_opt = optim.Adam(ps["ae"])
    disc_opt = optim.Adam(ps["disc"])

    for epoch in range(epochs):
        trainae_loss = 0
        traindisc_loss = 0
        for (x, _) in data:
            ae_opt.zero_grad()
            disc_opt.zero_grad()
            out = model(x)
            losses = ELBOWithDiscLoss(out["out"], out["logprior"], out["logpost"], x)
            losses["ae"].backward(retain_graph=True)
            losses["disc"].backward(retain_graph=True)
            ae_opt.step()
            disc_opt.step()
            trainae_loss += losses["ae"].item()
            traindisc_loss += losses["disc"].item()
        print(
            f"epoch {epoch},\tavg ae loss {trainae_loss/len(data.dataset)},\tavg disc loss {traindisc_loss/len(data.dataset)}"
        )
    return model


def test(model, data):
    model.eval()
    ae_res, disc_res = [], []
    for (x, _) in data:
        x = x.reshape(-1, 28 * 28)
        out = model(x)
        losses = ELBOWithDiscLoss(out["out"], out["logprior"], out["logpost"], x)
        # recon_x, prior, posterior = model(x)
        # disc_loss, ae_loss = ELBOWithDiscLoss(recon_x, prior, posterior, x)
        ae_res.append(losses["ae"])
        disc_res.append(losses["disc"])
    return ae_res, disc_res


def testaroulo():
    # model = AVB(
    #     Encoder(28 * 28, 128, 9),
    #     Discriminator(28 * 28, 128, 9),
    #     make_decoder(9, 128, 28 * 28),
    # )
    args = c.load_and_resolve("avb")
    for k, v in args["data"].items():
        args[f"{k}_data"] = v
    del args["data"]
    args["model"] = torch.jit.script(args["model"])
    model = train(args["model"], args["train_data"])

    return model, args["test_data"], test(args["model"], args["test_data"])


def test1(model, data):
    model.eval()
    res = []
    floss = torch.nn.MSELoss()
    for (x, _) in data:
        recon_x, _, _ = model(x)
        res.append(floss(recon_x, x).item())

    return res


def test2(model, data):
    model.eval()
    res = []
    floss = torch.nn.MSELoss()
    for (x, _) in data:
        out = model(x)["out"]
        res.append(floss(out, x).item())

    return msd(res)


if __name__ == "__main__":
    m, tdata, res = testaroulo()
    res1 = test2(m, tdata)
    print(msd(res1))
