from .MLP import MLP
import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(
        self,
        hparams: dict()
    ):
        super(VAE, self).__init__()
        self.hparams = hparams
        self.batch_norm = hparams["batch_norm"]
        self.Variational = hparams["Variational"]

        if self.Variational:
            self.encoder_sizes = [self.hparams["dim"]]+[self.hparams["encoder_width"]]* self.hparams["encoder_depth"]+ [self.hparams["emb_dim"]*2]
            self.decoder_sizes = [self.hparams["emb_dim"]]+[self.hparams["decoder_width"]]* self.hparams["decoder_depth"]+ [self.hparams["dim"]]
            self.encoder = MLP(self.encoder_sizes, batch_norm=self.batch_norm, last_layer_act="ReLU")
            self.decoder = MLP(self.decoder_sizes, batch_norm=self.batch_norm, last_layer_act="linear")

        else:
            self.encoder_sizes = [self.hparams["dim"]]+[self.hparams["encoder_width"]]* self.hparams["encoder_depth"]+ [self.hparams["emb_dim"]]
            self.decoder_sizes = [self.hparams["emb_dim"]]+[self.hparams["decoder_width"]]* self.hparams["decoder_depth"]+ [self.hparams["dim"]]
            self.encoder = MLP(self.encoder_sizes, batch_norm=self.batch_norm, last_layer_act="linear")
            self.decoder = MLP(self.decoder_sizes, batch_norm=self.batch_norm, last_layer_act="linear")

    def reparametrize(self, mu, sd):
        epsilon = torch.randn_like(sd)    
        z = mu + sd * epsilon 
        return z

    def get_emb(self, x):
        """
        get the embedding of given expression profiles of genes
        @param x: should be the shape [batch_size, hparams["dim]]
        """
        return self.encoder(x)[:, 0:self.hparams["emb_dim"]]
        
    def forward(self, x):
        """
        get the reconstruction of the expression profile of a gene
        @param x: should be the shape [batch_size, hparams["dim]]
        """
        latent = self.encoder(x)
        if self.Variational:
            mu = latent[:, 0:self.hparams["emb_dim"]]
            sd = latent[:, self.hparams["emb_dim"]:]
            assert mu.shape == sd.shape
            latent = self.reparametrize(mu, sd)
        reconstructed = self.decoder(latent)
        return reconstructed