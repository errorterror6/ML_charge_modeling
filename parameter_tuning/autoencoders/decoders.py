from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
import parameters

class DecoderBase(nn.Module, ABC):
    """
    Abstract base class for VAE decoders.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(DecoderBase, self).__init__()
        self.latent_dim = m['latent_dim']
        self.obs_dim = v['input_size']
        
    
    @abstractmethod
    def forward(self, z):
        """
        Forward pass through the decoder to produce reconstructed data.
        Args:
            z (Tensor): Latent vector sampled from latent distribution
            
        Returns:
            Tensor: Reconstructed data
        """
        pass


class MLPDecoder(DecoderBase):
    """
    MLP Decoder for VAE. Transforms latent variables to observation space.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(MLPDecoder, self).__init__(m, v)
        hidden_dim = m['nhidden']
        
        # Network architecture
        self.fc1 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.obs_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, z):
        """
        Forward pass decoding latent vector to observation space.
        
        Args:
            z (Tensor): Latent vector [batch_size, latent_dim]
            
        Returns:
            Tensor: Decoded observation [batch_size, obs_dim]
        """
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


class RNNDecoder(DecoderBase):
    """
    RNN Decoder for VAE. Generates sequences from latent variables.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(RNNDecoder, self).__init__(m, v)
        hidden_dim = v['rnn_nhidden']
        seq_len = 70  # Default sequence length, can be passed as parameter
        
        # Initial hidden state generator
        self.latent_to_hidden = nn.Linear(self.latent_dim, hidden_dim)
        
        # RNN cell
        self.rnn = nn.RNN(
            input_size=self.obs_dim,  # Previous output as input (full dimension)
            hidden_size=hidden_dim,
            nonlinearity='tanh',
            device=m['device'],
            batch_first=True
        )
        
        # Output projection
        self.hidden_to_output = nn.Linear(hidden_dim, self.obs_dim)
        
        # Store sequence length
        self.seq_len = seq_len
    
    def forward(self, z, seq_len=None):
        """
        Forward pass generating a sequence from latent vector.
        
        Args:
            z (Tensor): Latent vector [batch_size, latent_dim]
            seq_len (int, optional): Length of sequence to generate
            
        Returns:
            Tensor: Generated sequence [batch_size, seq_len, obs_dim]
        """
        batch_size = z.shape[0]
        seq_len = seq_len or self.seq_len
        device = z.device
        
        # Generate initial hidden state from latent
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch_size, hidden_dim]
        
        # Initialize first input (all zeros) with correct feature dimensions
        current_input = torch.zeros(batch_size, 1, self.obs_dim, device=device)
        
        # Store outputs
        outputs = []
        
        # Generate sequence autoregressively
        for t in range(seq_len):
            output, hidden = self.rnn(current_input, hidden)
            output = self.hidden_to_output(output)
            outputs.append(output)
            
            # Use prediction as next input
            current_input = output.detach()
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)  # [batch_size, seq_len, obs_dim]


class LSTMDecoder(DecoderBase):
    """
    LSTM Decoder for VAE. Generates sequences from latent variables with
    better handling of long-term dependencies.
    
    TODO: not reviewed.
    """
    def __init__(self, m=parameters.model_params, v=parameters.vae_params):
        super(LSTMDecoder, self).__init__(m, v)
        hidden_dim = v['rnn_nhidden']
        seq_len = 70  # Default sequence length, can be passed as parameter
        
        # Initial hidden state generator
        self.latent_to_hidden = nn.Linear(self.latent_dim, hidden_dim)
        self.latent_to_cell = nn.Linear(self.latent_dim, hidden_dim)
        
        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=self.obs_dim,  # Previous output as input (full dimension)
            hidden_size=hidden_dim,
            batch_first=True,
            device=m['device'],
        )
        
        # Output projection
        self.hidden_to_output = nn.Linear(hidden_dim, self.obs_dim)
        
        # Store sequence length
        self.seq_len = seq_len
    
    def forward(self, z, seq_len=None):
        """
        Forward pass generating a sequence from latent vector using LSTM.
        
        Args:
            z (Tensor): Latent vector [batch_size, latent_dim]
            seq_len (int, optional): Length of sequence to generate
            
        Returns:
            Tensor: Generated sequence [batch_size, seq_len, obs_dim]
        """
        batch_size = z.shape[0]
        seq_len = seq_len or self.seq_len
        device = z.device
        
        # Generate initial hidden and cell states from latent
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch_size, hidden_dim]
        cell = self.latent_to_cell(z).unsqueeze(0)      # [1, batch_size, hidden_dim]
        
        # Initialize first input (all zeros) with correct feature dimensions
        current_input = torch.zeros(batch_size, 1, self.obs_dim, device=device)
        
        # Store outputs
        outputs = []
        
        # Generate sequence autoregressively
        for t in range(seq_len):
            output, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            output = self.hidden_to_output(output)
            outputs.append(output)
            
            # Use prediction as next input
            current_input = output.detach()
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)  # [batch_size, seq_len, obs_dim]
