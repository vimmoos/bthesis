# def testaroulo_AE():
#     latent_dims = 2
#     autoencoder = Autoencoder(512, latent_dims, 784)
#     data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
#         './data', transform=torchvision.transforms.ToTensor(), download=True),
#                                        batch_size=512,
#                                        shuffle=True)

#     autoencoder = train(autoencoder, data)

#     plot_latent(autoencoder, data)
#     plt.savefig("autoencoder")
#     plt.clf()
#     plt.cla()
#     return autoencoder

# def testaroulo_VAE():
#     latent_dims = 2
#     vae = VAE(512, 784, latent_dims)
#     data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
#         './data', transform=torchvision.transforms.ToTensor(), download=True),
#                                        batch_size=512,
#                                        shuffle=True)

#     vae = train_VAE(vae, data)

#     plot_latent(vae, data)
#     plt.savefig("variational autoencoder")
#     plt.clf()
#     plt.cla()
#     return vae

# def testaroulo_AVB():

#     model = AVB(input=784, latent=2, hdec=512, henc=512, hdisc=512)

#     data = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
#         './data', transform=torchvision.transforms.ToTensor(), download=True),
#                                        batch_size=512,
#                                        shuffle=True)
#     m = train_AVB(model, data)

#     plot_latent_AVB(m, data)
#     plt.savefig("adversarial variational bayes")
#     plt.clf()
#     plt.cla()
#     return m
