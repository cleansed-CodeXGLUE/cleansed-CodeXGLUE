import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from .base_dl import BaseDeepLearningDetector
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.torch_utility import get_criterion_by_name


def vae_loss(x, x_recon, z_mu, z_logvar, beta=1.0, capacity=0.0):
    """Compute the loss of VAE

    Parameters
    ----------
    x : torch.Tensor, shape (n_samples, n_features)
        The input data.

    x_recon : torch.Tensor, shape (n_samples, n_features)
        The reconstructed data.

    z_mu : torch.Tensor, shape (n_samples, latent_dim)
        The mean of the latent distribution.

    z_logvar : torch.Tensor, shape (n_samples, latent_dim)
        The log variance of the latent distribution.

    beta : float, optional (default=1.0)
        The weight of KL divergence.

    capacity : float, optional (default=0.0)
        The maximum capacity of a loss bottleneck.

    Returns
    -------
    loss : torch.Tensor, shape (n_samples,)
        The loss of VAE.
    """
    # Reconstruction loss
    recon_loss = get_criterion_by_name('mse')(x_recon, x)

    # KL divergence
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - torch.exp(z_logvar),
                         dim=1), dim=0)
    kl_loss = torch.clamp(kl_loss, min=0, max=capacity)

    return recon_loss + beta * kl_loss


class VAE(BaseDeepLearningDetector):
    """ Variational auto encoder with Attention-based encoder and decoder """

    def __init__(self, contamination=0.1, preprocessing=True,
                 lr=1e-4, epoch_num=30, batch_size=32,
                 optimizer_name='adam',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 1e-5},
                 beta=1.0, capacity=0.0,
                 latent_dim=2,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False, dropout_rate=0.2,
                 nhead=1, num_encoder_layers=3, num_decoder_layers=3):
        super(VAE, self).__init__(contamination=0.1,
                                  preprocessing=True,
                                  lr=lr, epoch_num=epoch_num,
                                  batch_size=batch_size,
                                  optimizer_name=optimizer_name,
                                  loss_func=vae_loss,
                                  device=device, random_state=random_state,
                                  use_compile=use_compile,
                                  compile_mode=compile_mode,
                                  verbose=verbose,
                                  optimizer_params=optimizer_params)
        self.beta = beta
        self.capacity = capacity
        self.latent_dim = latent_dim
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Ensure latent_dim is divisible by nhead
        if self.latent_dim % self.nhead != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be divisible by nhead ({self.nhead})")

    def build_model(self):
        self.model = VAEModel(self.feature_size,
                              latent_dim=self.latent_dim,
                              hidden_activation_name=self.hidden_activation_name,
                              output_activation_name=self.output_activation_name,
                              batch_norm=self.batch_norm,
                              dropout_rate=self.dropout_rate,
                              nhead=self.nhead,
                              num_encoder_layers=self.num_encoder_layers,
                              num_decoder_layers=self.num_decoder_layers)

    def training_forward(self, batch_data):
        x = batch_data
        x = x.to(self.device)
        self.optimizer.zero_grad()
        x_recon, z_mu, z_logvar = self.model(x)
        loss = self.criterion(x, x_recon, z_mu, z_logvar,
                              beta=self.beta, capacity=self.capacity)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluating_forward(self, batch_data):
        x = batch_data
        x_gpu = x.to(self.device)
        x_recon, _, _ = self.model(x_gpu)

        # Check for NaNs in the reconstructed data
        if torch.isnan(x_recon).any():
            # print("NaN detected in x_recon during evaluation")
            x_recon[torch.isnan(x_recon)] = 0

        score = pairwise_distances_no_broadcast(x.numpy(),
                                                x_recon.cpu().numpy())
        return score


class VAEModel(nn.Module):
    def __init__(self,
                 feature_size,
                 latent_dim=2,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False, dropout_rate=0.2,
                 nhead=4, num_encoder_layers=3, num_decoder_layers=3):
        super(VAEModel, self).__init__()

        self.feature_size = feature_size
        self.latent_dim = latent_dim
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Ensure latent_dim is divisible by nhead
        if self.latent_dim % self.nhead != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be divisible by nhead ({self.nhead})")

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.encoder_mu = nn.Linear(feature_size, latent_dim)
        self.encoder_logvar = nn.Linear(feature_size, latent_dim)
        self.decoder_output = nn.Linear(latent_dim, feature_size)

    def _build_encoder(self):
        encoder_layers = TransformerEncoderLayer(
            d_model=self.feature_size, nhead=self.nhead, dropout=self.dropout_rate, batch_first=True)
        return TransformerEncoder(encoder_layers, num_layers=self.num_encoder_layers)

    def _build_decoder(self):
        decoder_layers = TransformerDecoderLayer(
            d_model=self.latent_dim, nhead=self.nhead, dropout=self.dropout_rate, batch_first=True)
        return TransformerDecoder(decoder_layers, num_layers=self.num_decoder_layers)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon, z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        # Ensure logvar is within a reasonable range
        logvar = torch.clamp(logvar, min=-20, max=20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(std.device)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x.unsqueeze(1))  # Add sequence dimension
        h = h.squeeze(1)  # Remove sequence dimension
        z_mu = self.encoder_mu(h)
        z_logvar = self.encoder_logvar(h)
        return z_mu, z_logvar

    def decode(self, z):
        h = self.decoder(z.unsqueeze(1), z.unsqueeze(1)
                         )  # Add sequence dimension
        h = h.squeeze(1)  # Remove sequence dimension
        x_recon = self.decoder_output(h)
        return x_recon
    
    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nhead, num_encoder_layers, num_decoder_layers, max_seq_length):
        super(TransformerVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Mean and log variance for latent space
        self.fc_mean = nn.Linear(hidden_dim * max_seq_length, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * max_seq_length, latent_dim)
        
        # Decoder
        decoder_layers = TransformerDecoderLayer(hidden_dim, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Linear layer to map latent space back to input space
        self.fc_out = nn.Linear(latent_dim, hidden_dim * max_seq_length)
        
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, hidden_dim))

    def encode(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, seq_length):
        z = self.fc_out(z)
        z = z.view(z.size(0), seq_length, -1)
        z = self.transformer_decoder(z, z)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mean, logvar

    def loss_function(self, recon_x, x, mean, logvar):
        BCE = F.cross_entropy(recon_x.view(-1, self.input_dim), x.view(-1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD

# Example usage
input_dim = 10000  # Vocabulary size for text data
hidden_dim = 512
latent_dim = 256
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
max_seq_length = 100

model = TransformerVAE(input_dim, hidden_dim, latent_dim, nhead, num_encoder_layers, num_decoder_layers, max_seq_length)

# Example input
x = torch.randint(0, input_dim, (32, max_seq_length))  # Batch size 32, sequence length 100

# Forward pass
recon_x, mean, logvar = model(x)

# Compute loss
loss = model.loss_function(recon_x, x, mean, logvar)
print(loss)

'''
