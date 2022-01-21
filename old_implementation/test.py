import sys

sys.path.append("/home/vimmoos/thesis/autoencoders/")

import vae
import avb1
import torchvision
import torch
import matplotlib.pyplot as plt

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def plot_latent(model, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def make_data():
    return (torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True),
                                        batch_size=512),
            torch.utils.data.DataLoader(torchvision.datasets.MNIST(
                './data',
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True),
                                        batch_size=512))


algos = {
    "AE": [vae.Autoencoder, vae.train],
    "VAE": [vae.VAE, vae.train_VAE],
    "AVB": [avb1.AVB, avb1.train_AVB]
}


def test():
    pass
