import torch
import torch.nn as nn

from GymExperiments.architectures.multihead import Dualhead


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar) # TODO why 0.5 ?
    eps = torch.randn_like(std)
    return mu + eps*std # TODO what?!


class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_dim, representation_dim):
        super().__init__()
        # TODO assert encoder output size is equal to encoder_dim
        # TODO assert decoder input size is equal to representation_dim

        self._encoder = encoder
        self.decoder = decoder

        fcmu = torch.nn.Linear(encoder_dim, representation_dim)
        fcvar = torch.nn.Linear(encoder_dim, representation_dim)

        self._vae_encoder = Dualhead(encoder,fcmu, fcvar)

    @property
    def encoder(self):
        return self._encoder

    @property
    def vae_encoder(self):
        return self._vae_encoder

    def encode(self, x):
        return self.vae_encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # TODO why 0.5 ?
        eps = torch.randn_like(std)
        return mu + eps*std # TODO what?!

    def forward(self, x):
        mu, logvar = self.encode(x)
        
        if self.train:
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
        else:
            return self.decode(mu), mu, logvar



def instaniate_SimpleCNNVAE(representation_dim, image_channels=3):
    from GymExperiments.architectures.cnn import SimpleCNNEncoder, SimpleCNNDecoder
    encoder_dim = 2*representation_dim
    encoder = SimpleCNNEncoder(encoder_dim, image_channels)
    decoder = SimpleCNNDecoder(representation_dim, image_channels)
    return VAE(encoder, decoder, encoder_dim, representation_dim)
