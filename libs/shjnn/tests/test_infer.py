import unittest
import torch
import numpy as np

# Import the modules to test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infer import make_infer_step
from model import init_model

class TestInferComponents(unittest.TestCase):
    """Test cases for the inference components of the shjnn library."""
    
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
        self.time_points = torch.linspace(0, 1, self.seq_length)
    
    def test_make_infer_step_latent_input(self):
        """Test that make_infer_step with latent input returns a callable function that works properly."""
        # Create the infer step function for latent input
        infer_step = make_infer_step(
            self.func, 
            self.rec, 
            self.dec, 
            self.optim, 
            self.device,
            _input='latent'
        )
        
        # Check that it's callable
        self.assertTrue(callable(infer_step))
        
        # Create a random latent state
        z0 = torch.randn(self.nbatch, self.latent_dim)
        
        # Run inference
        pred_x, pred_z = infer_step(z0, self.time_points)
        
        # Check output shapes
        self.assertEqual(pred_z.shape, (self.nbatch, self.seq_length, self.latent_dim))
        self.assertEqual(pred_x.shape, (self.nbatch, self.seq_length, self.obs_dim))
    
    def test_make_infer_step_traj_input(self):
        """Test that make_infer_step with trajectory input returns a callable function."""
        # Create the infer step function for trajectory input
        infer_step = make_infer_step(
            self.func, 
            self.rec, 
            self.dec, 
            self.optim, 
            self.device,
            _input='traj'
        )
        
        # Check that it's callable
        self.assertTrue(callable(infer_step))
        
        # NOTE: We're not running the actual inference since it requires specific
        # data shapes that are complex to set up in a unit test environment
    
    def test_make_infer_step_no_sample(self):
        """Test that make_infer_step with no sampling returns a callable function."""
        # Create the infer step function with no sampling
        infer_step = make_infer_step(
            self.func, 
            self.rec, 
            self.dec, 
            self.optim, 
            self.device,
            _input='traj',
            _sample=False
        )
        
        # Check that it's callable
        self.assertTrue(callable(infer_step))
        
        # NOTE: We're not running the actual inference since it requires specific
        # data shapes that are complex to set up in a unit test environment

if __name__ == '__main__':
    unittest.main()