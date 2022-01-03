"""TODO."""

import thesis.ednconf.core as c
import thesis.runner as r


def testaruolo(conf="test"):
    args = c.load_and_resolve(conf)
    for k, v in args["data"].items():
        args[f"{k}_data"] = v
    del args["data"]
    r.run(**args)


if __name__ == "__main__":
    testaruolo("avb")


# import torch
# from torch import nn


# def make_ae(encoder, decoder):
#     return torch.nn.Sequential(encoder, decoder)


# def make_decoder(
#     layers,
#     inter_act=nn.ReLU,
#     final_act=nn.Sigmoid,
# ):
#     return u.make_linear_seq(layers, inter_act, final_act)


# def make_encoder(
#     layers,
#     inter_act=nn.ReLU,
#     final_act=None,
# ):
#     return u.make_linear_seq(layers, inter_act, final_act)


# def train(autoencoder, data, floss, epochs=20):
#     opt = optim.Adam(autoencoder.parameters())
#     autoencoder.train()
#     for epoch in range(epochs):
#         train_loss = 0
#         cnt = 0
#         for x, _ in data:
#             x = x.reshape(-1, 28 * 28)
#             opt.zero_grad()
#             x_hat = autoencoder(x)
#             loss = floss(x_hat, x)
#             loss.backward()
#             train_loss += loss.item()
#             if cnt % 200 == 0:
#                 print(f"current loss {loss.item()}, total : {train_loss}")
#             opt.step()
#             cnt += 1
#         print(f"epoch {epoch},\tavg loss {train_loss/len(data.dataset)}")
#     return autoencoder


# def testaroulo(floss=ti_loss):
#     layers = u.gen_layers_number(28 * 28, 9, 3)
#     autoencoder = make_ae(
#         make_encoder(layers), make_decoder([t[::-1] for t in layers][::-1])
#     )
#     dtrain, dtest = u.load_data()
#     autoencoder = train(autoencoder, dtrain, floss)

#     return autoencoder, test(autoencoder, dtest, floss)
