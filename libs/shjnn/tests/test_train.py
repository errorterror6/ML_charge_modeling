import unittest
import torch
import numpy as np

# Import the modules to test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import make_train_step, log_normal_pdf, normal_kl
from model import init_model

class TestTrainComponents(unittest.TestCase):
    """Test cases for the training components of the shjnn library."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Model parameters
        self.latent_dim = 4
        self.nhidden = 20
        self.rnn_nhidden = 25
        self.obs_dim = 6
        self.nbatch = 5
        self.lr = 0.01
        self.device = torch.device('cpu')
        
        # Initialize model components
        self.func, self.rec, self.dec, self.optim, self.device = init_model(
            self.latent_dim, 
            self.nhidden, 
            self.rnn_nhidden, 
            self.obs_dim, 
            self.nbatch, 
            self.lr, 
            device=self.device
        )
        
        # Create synthetic data for testing
        self.seq_length = 8
        
        # Create a mini-batch of trajectories (batch, sequence, features)
        self.traj_batch = torch.randn(self.nbatch, self.seq_length, self.obs_dim)
        
        # Create a mini-batch of time points (batch, sequence, 1)
        self.time_batch = torch.linspace(0, 1, self.seq_length).unsqueeze(0).unsqueeze(-1)
        self.time_batch = self.time_batch.repeat(self.nbatch, 1, 1)
    
    def test_log_normal_pdf(self):
        """Test the log normal PDF calculation."""
        # Create test inputs
        x = torch.randn(10, 5)
        mean = torch.zeros(10, 5)
        logvar = torch.zeros(10, 5)
        
        # Calculate log PDF
        log_pdf = log_normal_pdf(x, mean, logvar)
        
        # Check shape
        self.assertEqual(log_pdf.shape, x.shape)
        
        # For mean=0, logvar=0 (std=1), the log PDF should follow a specific pattern
        expected_log_pdf = -0.5 * (np.log(2 * np.pi) + x**2)
        self.assertTrue(torch.allclose(log_pdf, expected_log_pdf, atol=1e-5))
    
    def test_normal_kl(self):
        """Test the KL divergence calculation."""
        # Create test inputs for two normal distributions
        mu1 = torch.randn(10, 5)
        lv1 = torch.zeros(10, 5)  # log variance of 0 (std=1)
        mu2 = torch.zeros(10, 5)
        lv2 = torch.zeros(10, 5)  # log variance of 0 (std=1)
        
        # Calculate KL divergence
        kl = normal_kl(mu1, lv1, mu2, lv2)
        
        # Check shape
        self.assertEqual(kl.shape, mu1.shape)
        
        # For equal variances (std=1), the KL divergence should be 0.5 * ||mu1 - mu2||^2
        expected_kl = 0.5 * (mu1 - mu2)**2
        self.assertTrue(torch.allclose(kl, expected_kl, atol=1e-5))
    
    def test_make_train_step(self):
        """Test that make_train_step returns a callable function."""
        # Create the train step function
        train_step = make_train_step(
            self.func, 
            self.rec, 
            self.dec, 
            self.optim, 
            self.device
        )
        
        # Check that it's callable
        self.assertTrue(callable(train_step))

if __name__ == '__main__':
    unittest.main()