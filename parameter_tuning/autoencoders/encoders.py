from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import sys
sys.path.append('..')
import parameters

class EncoderBase(nn.Module, ABC):
    """
    Abstract base class for VAE encoders.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(EncoderBase, self).__init__()
        self.latent_dim = m['latent_dim']
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the encoder to produce latent distribution parameters.
        Returns: mu, log_var
        """
        pass


class MLPEncoder(EncoderBase):
    """
    MLP Encoder for VAE that transforms input data to latent space.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(MLPEncoder, self).__init__(m, v)
        hidden_dim = m['nhidden']
        
        # Network architecture
        self.fc1 = nn.Linear(m['obs_dim'], hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, self.latent_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass encoding data to latent distribution parameters.
        
        Args:
            x (Tensor): Input data [batch_size, obs_dim]
            
        Returns:
            tuple: (mu, log_var) latent distribution parameters
        """
        # Flatten sequence data if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            
        out = self.fc1(x)
        out = self.relu(out)
        
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)
        
        return mu, log_var
    
    
class RNNEncoder(EncoderBase):
    """
    RNN Encoder for VAE.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(RNNEncoder, self).__init__(m, v)
        hidden_dim = v['rnn_nhidden']
        latent_dim = m['latent_dim']
        self.rnn = nn.RNN(
            input_size=v['input_size'],
            hidden_size=hidden_dim,
            # use relu + clip_gradient if poor results with tanh
            nonlinearity='tanh',
            batch_first=True
            )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        outputs, h_n = self.rnn(x)
        final_hidden_state = outputs[:, -1, :]
        
        # Get latent distribution parameters
        mu = self.fc_mu(final_hidden_state)
        log_var = self.fc_log_var(final_hidden_state)
        
        return mu, log_var
    

class LSTMEncoder(EncoderBase):
    """
    LSTM Encoder for VAE.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(LSTMEncoder, self).__init__()
        hidden_dim = v['rnn_nhidden']
        latent_dim = m['latent_dim']
        self.lstm = nn.LSTM(
            input_size=v['input_size'],
            hidden_size=hidden_dim,
            batch_first=True
            )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        final_hidden_state = outputs[:, -1, :]
        
        # Get latent distribution parameters
        mu = self.fc_mu(final_hidden_state)
        log_var = self.fc_log_var(final_hidden_state)
        
        return mu, log_var
