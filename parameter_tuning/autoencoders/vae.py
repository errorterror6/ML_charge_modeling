import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import encoders
import decoders

from abc import ABC, abstractmethod

import sys
sys.path.append('..')
import parameters
import loader




class VAE(nn.Module):
    def __init__(self, enc, dec, loss_function=None):
        super(VAE, self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.loss_fn = torch.nn.MSELoss() if loss_function is None else loss_function
        
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, x):
        """
        Input x should be whole sequence 
        of shape [batch_size, sequence_length, feature_dim]
        expected [x, 70, 6].
        """
        mu, log_var = self.encoder.forward(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var, z
    
    def loss_function(self, traj, reconstruction, mu, log_var, loss_fn, beta=1.0):
        recon_loss = self.loss_fn(reconstruction, traj)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * KLD, recon_loss, KLD
    
    def train_nepochs(self, n_epochs, m=parameters.model_params, v=parameters.vae_params, r=parameters.records):
        train, val, orig = loader.get_formatted_data()
        train = train.to(m['device'])
        val = val.to(m['device'])
        orig = orig.to(m['device'])
        total_loss_history = []
        recon_loss_history = []
        kl_loss_history = []
        print(f"logs: vae: train_nepochs: training for {n_epochs} epochs.")
        for epoch in range(n_epochs):
            total_loss = 0
            recon_loss = 0
            kl_loss = 0
            
            try:
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                for batch_idx, (x, y, meta) in enumerate(train):
                    
                    x = torch.squeeze(x)
                    y = torch.squeeze(y)
                    obs = torch.cat((x, y), dim=-1)
                    obs = torch.cat((obs, meta), dim=-1)
                    obs = torch.unsqueeze(obs, dim=2)
                    print("debug: vae: train_nepochs: obs shape: ", obs.shape)
            except KeyboardInterrupt:
                print("logs: vae: train_nepochs: Training interrupted by keyboard.")
                exit(1)
                
        
    @classmethod
    def create(cls, m=parameters.model_params, v=parameters.vae_params):
        """
        Create VAE model with encoder and decoder.
        
        Args:
            m (dict): Model parameters
            v (dict): VAE parameters
        
        Returns:
            VAE: Instance of VAE model
        """
        if parameters.trainer == 'RNN-VAE':
            encoder = encoders.RNNEncoder(m, v)
            decoder = decoders.RNNDecoder(m, v)
        elif parameters.trainer == 'MLP-VAE':
            encoder = encoders.MLPEncoder(m, v)
            decoder = decoders.MLPDecoder(m, v)
        elif parameters.trainer == 'LSTM-VAE':
            encoder = encoders.LSTMEncoder(m, v)
            decoder = decoders.LSTMDecoder(m, v)
        else:
            print(f"auto: vae: Error: Unknown trainer type '{parameters.trainer}'.")
            exit(1)
        
        return cls(encoder, decoder)