# def plot_latent_AVB(model, data, num_batches=100):
#     recon = []
#     for i, (x, y) in enumerate(data):
#         x = torch.reshape(torch.squeeze(x), (512, 784))
#         z = model.encoder(x)
#         recon_liklihood = -torch.nn.functional.binary_cross_entropy(
#             model.decoder(z), x) * x.data.shape[0]
#         recon += [recon_liklihood]
#         # print(recon_liklihood)
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
#     print("====== MEAN =========")
#     print(sum(recon) / len(recon))

# # def plot_latent(autoencoder, data, num_batches=100):
# #     for i, (x, y) in enumerate(data):
# #         z = autoencoder.encoder(x)
# #         z = z.to('cpu').detach().numpy()
# #         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
# #         if i > num_batches:
# #             plt.colorbar()
# #             break
# def plot_latent(model, data, num_batches=100):
#     recon = []
#     for i, (x, y) in enumerate(data):
#         z = model.encoder(x)
#         x = torch.reshape(torch.squeeze(x), (512, 784))
#         recon_liklihood = -torch.nn.functional.binary_cross_entropy(
#             torch.reshape(torch.squeeze(model.decoder(z)),
#                           (512, 784)), x) * x.data.shape[0]
#         recon += [recon_liklihood]
#         # print(recon_liklihood)
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
#     print("====== MEAN =========")
#     print(sum(recon) / len(recon))

# def plot_latent(autoencoder: Autoencoder, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         z = autoencoder.encoder(x)
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
