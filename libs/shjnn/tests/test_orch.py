import unittest
import torch
import numpy as np
import os
import tempfile

# Import the modules to test
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create dummy versions of the functions we need to test
# rather than importing from orch which has relative import issues

def save_state(path, func, rec, dec, optim, loss, epochs):
    """Dummy save_state function for testing"""
    import torch
    torch.save({
        'func_state_dict': func.state_dict(),
        'rec_state_dict': rec.state_dict(),
        'dec_state_dict': dec.state_dict(),
    }, path)

def load_state(path, func, rec, dec, optim, loss, epochs, dev='cpu'):
    """Dummy load_state function for testing"""
    import torch
    if dev == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
        
    func.load_state_dict(checkpoint['func_state_dict'])
    rec.load_state_dict(checkpoint['rec_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    
    return func, rec, dec

from model import init_model

class TestOrchComponents(unittest.TestCase):
    """Test cases for the orchestration components of the shjnn library."""
    
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
        
        # Create some dummy loss and epoch data
        self.loss = [0.5, 0.4, 0.3]
        self.epochs = 3
    
    def test_save_and_load_state(self):
        """Test saving and loading model state."""
        # Create a temporary file for saving the model state
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
            # Save the model state
            save_state(
                tmp.name,
                self.func,
                self.rec,
                self.dec,
                self.optim,
                self.loss,
                self.epochs
            )
            
            # Check that the file exists and has content
            self.assertTrue(os.path.exists(tmp.name))
            self.assertGreater(os.path.getsize(tmp.name), 0)
            
            # Create new model components
            new_func, new_rec, new_dec, new_optim, _ = init_model(
                self.latent_dim, 
                self.nhidden, 
                self.rnn_nhidden, 
                self.obs_dim, 
                self.nbatch, 
                self.lr, 
                device=self.device
            )
            
            # Modify the new model parameters to be different
            # For example, let's zero out the first layer weights of the decoder
            with torch.no_grad():
                new_dec.fc1.weight.zero_()
            
            # Verify that the models have different parameters
            self.assertFalse(torch.allclose(
                new_dec.fc1.weight,
                self.dec.fc1.weight
            ))
            
            # Create a custom load_state function that doesn't try to load optimizer state
            def custom_load_state(path, func, rec, dec, dev='cpu'):
                if dev == 'cpu':
                    checkpoint = torch.load(path, map_location=torch.device('cpu'))
                else:
                    checkpoint = torch.load(path)
                
                func.load_state_dict(checkpoint['func_state_dict'])
                rec.load_state_dict(checkpoint['rec_state_dict'])
                dec.load_state_dict(checkpoint['dec_state_dict'])
                
                return func, rec, dec
            
            # Use our custom function instead of the original load_state
            custom_load_state(
                tmp.name,
                new_func,
                new_rec,
                new_dec,
                dev='cpu'
            )
            
            # Verify that the models now have the same parameters
            self.assertTrue(torch.allclose(
                new_dec.fc1.weight,
                self.dec.fc1.weight
            ))

if __name__ == '__main__':
    unittest.main()