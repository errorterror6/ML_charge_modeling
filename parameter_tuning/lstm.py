import parameters
from torch import nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import loader
import sys
sys.path.append('../libs/')
import shjnn
from rnn import RNN

class LSTM(RNN):
    """
    inherits from RNN, but substitutes the RNN layer with an LSTM layer
    """
    def __init__(self, m=parameters.model_params):
        # Call parent constructor
        super().__init__(m)
        
        # Override the RNN with LSTM
        self.temporal = nn.LSTM(
            input_size=2,
            hidden_size=m['rnn_nhidden'],
            device=m['device'],
            batch_first=True
        )
        
        # We'll store the cell state here during sequence processing
        self.cell_state = None
        
        # Reinitialize optimizer with the new parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=m['lr'])
        
        # Reset visualiser for this instance
        self.visualiser = self.Visualiser(self)
    
    def init_hidden(self, batch_size):
        # LSTM needs both hidden state and cell state
        d = 1  # num_layers * num_directions
        h0 = torch.zeros(d, batch_size, self.model_params['nhidden'], device=self.model_params['device'])
        c0 = torch.zeros(d, batch_size, self.model_params['nhidden'], device=self.model_params['device'])
        # Store the cell state as an instance variable
        self.cell_state = c0
        # Return only h0 to match parent RNN's interface
        return h0
    
    def forward(self, data, hidden):
        # For LSTM, we need (h0, c0) as hidden state
        # Use the stored cell state or initialize if None
        if self.cell_state is None:
            batch_size = hidden.size(1)
            self.cell_state = torch.zeros_like(hidden)
        
        # LSTM returns output and tuple of (h_n, c_n)
        _, (h_t, c_t) = self.temporal(data, (hidden, self.cell_state))
        
        # Update our stored cell state
        #TODO: check out what cell_state does in LSTMs and how its meant to be used.
        
        self.cell_state = c_t
        
        h2 = self.h2h1(h_t)
        output = self.h2o(h2)
        
        # Return output and hidden state (h_t only to match parent class interface)
        return output, h_t
        
    # Override forward_step to reset cell state at the beginning of each sequence
    def forward_step(self, obs, train=True):
        # Reset cell state at the beginning of processing a new sequence
        self.cell_state = None
        # Call parent method
        return super().forward_step(obs, train)