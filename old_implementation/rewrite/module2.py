import module as k

k.mona
k.nonmi

# setattr(sys.modules[module.__name__], '__getattribute__',
#         module.__getattribute__)

# module.a

# getattr(module, 'a')
# traceback.print_stack(file=sys.stdout)


def structure(struct):
    def inner():
        for _, vs in struct().items():
            for v in vs[1:]:
                print(v)
        return struct

    return inner()


def network (**kwargs):
    


encoder = {
    k.input: (k.Linear, k.input, k.ehid),
    k.output: (k.Linear, k.ehid, k.latent)
}

decoder = {
    k.input: (k.Linear, k.latent, k.dhid),
    k.output: (k.Linear, k.dhid, k.output)
}

autoencoder = {
    k.encoder: (encoder, k.input, k.ehid, k.latent),
    k.decoder: (decoder, k.latent, k.dhid, k.output)
}
