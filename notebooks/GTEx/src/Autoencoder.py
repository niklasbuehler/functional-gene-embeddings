import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size, n_hidden=512):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, encoding_size),
            nn.ReLU()   
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x