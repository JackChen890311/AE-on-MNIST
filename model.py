import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_dims):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_layers = self.construct(input_size, hidden_dims, latent_size)
        self.encoder = nn.Sequential(*encoder_layers)
            
        # Decoder
        hidden_dims.reverse()
        decoder_layers = self.construct(latent_size, hidden_dims, input_size)
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def construct(self, start, hidden, end):
        layers = []
        in_size = start
        for h_dim in hidden:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_size, h_dim),
                    nn.ReLU(True)
                )
            )
            in_size = h_dim
        layers.append(nn.Linear(in_size, end))
        return layers

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ =='__main__':
    from constant import CONSTANT
    C = CONSTANT()

    hidden_dims = [2048, 512, 128]
    model = Autoencoder(C.in_size, C.latent_size, C.hidden_dims)
    print(model)
    dummy = torch.rand((1,C.in_size))
    latent = model.encoder(dummy)
    result = model.decoder(latent)
    print(latent.shape, result.shape)