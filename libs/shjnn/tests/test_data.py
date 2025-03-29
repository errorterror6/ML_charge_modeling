import unittest
import torch
import numpy as np

# Import the module to test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import CustomDataset

class TestDataComponents(unittest.TestCase):
    """Test cases for the data handling components of the shjnn library."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data for testing
        self.n_samples = 10
        self.feature_dim = 6
        self.seq_length = 8
        
        # Create trajectories (batches, sequence length, features)
        self.trajs = [torch.randn(self.seq_length, self.feature_dim) for _ in range(self.n_samples)]
        
        # Create time points (batches, sequence length, 1)
        self.times = [torch.linspace(0, 1, self.seq_length).unsqueeze(1) for _ in range(self.n_samples)]
        
    def test_custom_dataset(self):
        """Test the CustomDataset implementation."""
        # Initialize the dataset
        dataset = CustomDataset(self.trajs, self.times)
        
        # Test the length of the dataset
        self.assertEqual(len(dataset), self.n_samples)
        
        # Test getting an item from the dataset
        x, y = dataset[0]
        
        # Check that the returned item has the correct shape
        self.assertEqual(x.shape, self.trajs[0].shape)
        self.assertEqual(y.shape, self.times[0].shape)
        
        # Test iterating through the dataset
        for i, (x, y) in enumerate(dataset):
            self.assertEqual(x.shape, self.trajs[i].shape)
            self.assertEqual(y.shape, self.times[i].shape)
            
            # Check that the data matches
            self.assertTrue(torch.allclose(x, self.trajs[i]))
            self.assertTrue(torch.allclose(y, self.times[i]))

if __name__ == '__main__':
    unittest.main()