"""TODO."""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import thesis.trainers as tr
import thesis.utils as u
import thesis.vanilla as va

ti_loss = torch.nn.MSELoss()
trans = transforms.ToTensor()
train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Lambda(lambda x: trans(x).reshape(-1, 28 * 28)),
)


def mse(out, x):
    return {"ae": ti_loss(out, x)}


def test(n=3):
    layers = u.gen_layers_number(28 * 28, 9, n)
    model = va.VanillaAutoencoder(layers, [t[::-1] for t in layers][::-1])

    trainer = tr.Trainer(
        **{
            "ae": {
                "optim": optim.Adam,
            },
        }
    )
    train, test = u.load_data()

    trainer.train(model, train, mse)


if __name__ == "__main__":
    for x in range(3, 10):
        print(f"========={x}=========")
        test(x)

# Loss ae: 0.01574177460173766
# current loss 0.01603039726614952, total : 28.32065551355481

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
