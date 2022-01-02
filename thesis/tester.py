# def test(model, data, floss):
#     model.eval()
#     return [
#         floss(model(x.reshape(-1, 28 * 28)), x.reshape(-1, 28 * 28)).item()
#         for x, _ in data
#     ]


# def test(model, data):
#     model.eval()
#     return [
#         ELBOLoss(*model(x.reshape(-1, 28 * 28)),
#                  x.reshape(-1, 28 * 28)).item() for x, _ in data
#     ]


# def test(model, data):
#     model.eval()
#     ae_res, disc_res = [], []
#     for (x, _) in data:
#         x = x.reshape(-1, 28 * 28)
#         recon_x, prior, posterior = model(x)
#         disc_loss, ae_loss = ELBOWithDiscLoss(recon_x, prior, posterior, x)
#         ae_res.append(ae_loss)
#         disc_res.append(disc_loss)
#     return ae_res, disc_res
