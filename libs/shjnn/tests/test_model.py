import unittest
import torch
import numpy as np

# Import the module to test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import init_model, RecognitionRNN, LatentODEfunc, Decoder

class TestModelComponents(unittest.TestCase):
    """Test cases for the model components of the shjnn library."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latent_dim = 4
        self.nhidden = 20
        self.rnn_nhidden = 25
        self.obs_dim = 6
        self.nbatch = 10
        self.lr = 0.01
        self.device = torch.device('cpu')
        
    def test_init_model(self):
        """Test that init_model correctly initializes all model components."""
        func, rec, dec, optim, device = init_model(
            self.latent_dim, 
            self.nhidden, 
            self.rnn_nhidden, 
            self.obs_dim, 
            self.nbatch, 
            self.lr, 
            device=self.device
        )
        
        # Test that all components are correctly initialized
        self.assertIsInstance(func, LatentODEfunc)
        self.assertIsInstance(rec, RecognitionRNN)
        self.assertIsInstance(dec, Decoder)
        self.assertIsInstance(optim, torch.optim.Adam)
        self.assertEqual(device, self.device)
        
        # Test that the models are on the correct device
        self.assertEqual(next(func.parameters()).device, self.device)
        self.assertEqual(next(rec.parameters()).device, self.device)
        self.assertEqual(next(dec.parameters()).device, self.device)
        
    def test_recognition_rnn(self):
        """Test the RecognitionRNN implementation."""
        rec = RecognitionRNN(
            latent_dim=self.latent_dim, 
            obs_dim=self.obs_dim, 
            hidden_size=self.rnn_nhidden, 
            batch_size=self.nbatch
        )
        
        # Test initialization of hidden state
        h = rec.initHidden()
        self.assertEqual(h.shape, (self.nbatch, self.rnn_nhidden))
        self.assertEqual(h.sum().item(), 0.0)  # Should be initialized with zeros
        
        # Test forward pass
        x = torch.randn(self.nbatch, self.obs_dim)
        out, h_new = rec.forward(x, h)
        
        # Output should have twice the latent dimension (mean and logvar)
        self.assertEqual(out.shape, (self.nbatch, self.latent_dim * 2))
        
        # New hidden state should have the correct shape
        self.assertEqual(h_new.shape, (self.nbatch, self.rnn_nhidden))
        
    def test_latent_ode_func(self):
        """Test the LatentODEfunc implementation."""
        func = LatentODEfunc(
            latent_dim=self.latent_dim, 
            hidden_size=self.nhidden
        )
        
        # Test forward pass
        t = torch.tensor(0.0)  # Time variable (not used in this implementation)
        x = torch.randn(self.nbatch, self.latent_dim)
        out = func.forward(t, x)
        
        # Output should have the same shape as input for the latent dimension
        self.assertEqual(out.shape, (self.nbatch, self.latent_dim))
        
        # Test that NFE (number of function evaluations) is incremented
        self.assertEqual(func.nfe, 1)
        
    def test_decoder(self):
        """Test the Decoder implementation."""
        dec = Decoder(
            latent_dim=self.latent_dim, 
            obs_dim=self.obs_dim, 
            hidden_size=self.nhidden
        )
        
        # Test forward pass
        z = torch.randn(self.nbatch, self.latent_dim)
        out = dec.forward(z)
        
        # Output should have the observational dimension
        self.assertEqual(out.shape, (self.nbatch, self.obs_dim))

if __name__ == '__main__':
    unittest.main()