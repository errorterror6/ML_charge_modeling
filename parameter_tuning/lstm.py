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
        
        # Reinitialize optimizer with the new parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=m['lr'])
        
        # Reset visualiser for this instance
        self.visualiser = self.Visualiser(self)
    
    def init_hidden(self, batch_size):
        # LSTM needs both hidden state and cell state
        d = 1  # num_layers * num_directions
        h0 = torch.zeros(d, batch_size, self.model_params['nhidden'])
        c0 = torch.zeros(d, batch_size, self.model_params['nhidden'])
        return (h0, c0)
    
    def forward(self, data, hidden):
        # LSTM returns output and tuple of (h_n, c_n)
        _, (h_t, c_t) = self.temporal(data, hidden)
        h2 = self.h2h1(h_t)
        output = self.h2o(h2)
        
        # Return output and hidden state (h_t only, to match parent class interface)
        return output, h_t