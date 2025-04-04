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
    
    def loss_function(self, traj, reconstruction, mu, log_var, beta=1.0):
        recon_loss = self.loss_fn(reconstruction, traj)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * KLD, recon_loss, KLD
    
    
    
    

    def train_step(self, input_data):
        #input data is expected to be [16, 70, 6]
        self.encoder.train()
        self.decoder.train()
        
        reversed_data = loader.reverse_traj(input_data)
        reconstruction, mu, log_var, z = self.forward(reversed_data)
        loss, recon_loss, kl_loss = self.loss_function(reversed_data, reconstruction, mu, log_var)
        self.optimizer.zero_grad()
        loss.backward()
        self.encoder.optimizer.step()
        
        return loss, recon_loss, kl_loss
    
    def eval_loss_fn(self, traj, reconstruction):
        # traj is expected to be [16, 70, 6], same as reconstruction
        #with features [x, t, y, intensity, bias, delay]
        
        #modify to be [16, 70, 2] to only have [x, t] features
        traj = traj[:, :, :2]
        reconstruction = reconstruction[:, :, :2]
        recon_loss = self.loss_fn(reconstruction, traj)
        return recon_loss
    
    def eval_step(self, input_data):
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            reversed_data = loader.reverse_traj(input_data)
            reconstruction, mu, log_var, z = self.forward(reversed_data)
            total_loss, feat_loss, kl_loss = self.loss_function(reversed_data, reconstruction, mu, log_var)
            eval_loss = self.eval_loss_fn(reversed_data, reconstruction) 
                  
        return eval_loss, (total_loss, feat_loss, kl_loss)
            
            
            
            
    
    def train_nepochs(self, n_epochs, m=parameters.model_params, v=parameters.vae_params, r=parameters.records):
        train, val, orig = loader.get_formatted_data()
        train = train.to(m['device'])
        val = val.to(m['device'])
        orig = orig.to(m['device'])
        total_loss_history = []
        recon_loss_history = []
        kl_loss_history = []
        eval_loss_history = []
        
        print(f"logs: vae: train_nepochs: training for {n_epochs} epochs.")
        for epoch in range(n_epochs):
            total_loss = 0
            recon_loss = 0
            kl_loss = 0
            eval_loss = 0
            
            try:
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                
                #send each bath to training.
                for batch_idx, (x, y, meta) in enumerate(train):
                    input_data = loader.compile_stacked_data(x, y, meta)
                    self.train_step(input_data)
                    
                #TODO: eval each batch and get the loss. record.
                for batch_idx, (x, y, meta) in enumerate(val):
                    input_data = loader.compile_stacked_data(x, y, meta)
                    eval_loss, loss = self.eval_step(input_data)
                    total_loss += loss[0]
                    recon_loss += loss[1]
                    kl_loss += loss[2]
                    eval_loss += eval_loss
                total_loss /= len(val)
                recon_loss /= len(val)
                kl_loss /= len(val)
                eval_loss /= len(val)
                total_loss_history.append(total_loss)
                recon_loss_history.append(recon_loss)
                kl_loss_history.append(kl_loss)
                eval_loss_history.append(eval_loss)
                print(f"Epoch {epoch}: Eval Loss {eval_loss:.3f}, "
                    f"KL Loss {kl_loss:.2f}, Feature Loss {recon_loss:.2f}, Total Loss {total_loss:.2f}")
                return n_epochs, eval_loss_history, (total_loss_history, recon_loss_history, kl_loss_history)
                
                    
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
            encoder = encoders.RNNEncoder(m, v).to(m['device'])
            decoder = decoders.RNNDecoder(m, v).to(m['device'])
        elif parameters.trainer == 'MLP-VAE':
            encoder = encoders.MLPEncoder(m, v).to(m['device'])
            decoder = decoders.MLPDecoder(m, v).to(m['device'])
        elif parameters.trainer == 'LSTM-VAE':
            encoder = encoders.LSTMEncoder(m, v).to(m['device'])
            decoder = decoders.LSTMDecoder(m, v).to(m['device'])
        else:
            print(f"auto: vae: Error: Unknown trainer type '{parameters.trainer}'.")
            exit(1)
        
        return cls(encoder, decoder)