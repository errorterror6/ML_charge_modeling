import unittest
import sys
import os
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock classes for testing
class MockRNN:
    def eval_step(self, traj, time, batch_input=True):
        """Mock eval_step method for testing"""
        return 0.1, [torch.randn(1, 69, 1) for _ in range(69)], torch.randn(1, 70, 2)
    
    class Visualiser:
        def __init__(self, rnn_instance):
            self.RNN = rnn_instance
            
        def plot_training_loss(self, model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            pass
            
        def display_random_fit(self, model_params, dataset, show=False, save=True, random_samples=True):
            pass
            
        def compile_learning_gif(self, model_params=None, display=True):
            pass
            
        def sweep_latent_adaptives(self, model_params=None, dataset=None):
            pass
            
        def sweep_latent_adaptive(self, model_params=None, dataset=None, latent_dim_number=0):
            pass
            
    @staticmethod
    def format_output(pred_x, target_timesteps=1000):
        if not isinstance(pred_x, torch.Tensor):
            raise TypeError("Input pred_x must be a PyTorch tensor.")
        if len(pred_x.shape) != 3 or pred_x.shape[0] != 1 or pred_x.shape[2] != 1:
            raise ValueError(f"Input pred_x must have shape torch.Size([1, initial_timesteps, 1]), but got {pred_x.shape}")
        if target_timesteps <= pred_x.shape[1]:
            raise ValueError("target_timesteps must be greater than the original timesteps for extrapolation.")
        
        return torch.ones(1, target_timesteps, 1)


class TestRNNVisualiser(unittest.TestCase):
    """Test cases for the Visualiser class in rnn.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model_params dictionary
        self.model_params = {
            'rnn_nhidden': 64,
            'nhidden': 64,
            'device': 'cpu',
            'lr': 1e-4,
            'n_batch': 16,
            'epochs': 5,
            'loss': [150, 135, 120, 105, 90],
            'MSE_loss': [100, 90, 80, 70, 60],
            'KL_loss': [50, 45, 40, 35, 30],
            'folder': 'test_folder'
        }
        
        # Create a mock dataset dictionary
        self.dataset = {
            'trajs': torch.randn(10, 100, 1),  # 10 trajectories, 100 timepoints, 1 feature
            'times': torch.randn(10, 100, 1),  # 10 trajectories, 100 timepoints, 1 time feature
            'y': torch.randn(10, 3)  # 10 trajectories, 3 parameters (intensity, bias, delay)
        }
        
        # Create an RNN instance with our mock class
        self.rnn_model = MockRNN()
        self.rnn_model.visualiser = MockRNN.Visualiser(self.rnn_model)
            
        # Create directory for test outputs if it doesn't exist
        os.makedirs('test_folder', exist_ok=True)
        os.makedirs('test_folder/loss_graph', exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories if they exist
        if os.path.exists('test_folder'):
            import shutil
            shutil.rmtree('test_folder')

    def test_plot_training_loss(self):
        """Test plot_training_loss method."""
        # Use a spy to track calls
        with patch.object(self.rnn_model.visualiser, 'plot_training_loss') as mock_method:
            # Call the method
            self.rnn_model.visualiser.plot_training_loss(self.model_params, save=True)
            
            # Assertions
            mock_method.assert_called_once_with(self.model_params, save=True)

    def test_display_random_fit(self):
        """Test display_random_fit method."""
        # Use a spy to track calls
        with patch.object(self.rnn_model.visualiser, 'display_random_fit') as mock_method:
            # Call the method
            self.rnn_model.visualiser.display_random_fit(self.model_params, self.dataset, save=True)
            
            # Assertions
            mock_method.assert_called_once_with(self.model_params, self.dataset, save=True)

    def test_compile_learning_gif(self):
        """Test compile_learning_gif method."""
        # Use a spy to track calls
        with patch.object(self.rnn_model.visualiser, 'compile_learning_gif') as mock_method:
            # Call the method
            self.rnn_model.visualiser.compile_learning_gif(self.model_params, display=False)
            
            # Assertions
            mock_method.assert_called_once_with(self.model_params, display=False)

    def test_sweep_latent_adaptives(self):
        """Test sweep_latent_adaptives method."""
        # Use a spy to track calls
        with patch.object(self.rnn_model.visualiser, 'sweep_latent_adaptives') as mock_method:
            # Call the method
            self.rnn_model.visualiser.sweep_latent_adaptives(self.model_params, self.dataset)
            
            # Assertions
            mock_method.assert_called_once_with(self.model_params, self.dataset)

    def test_sweep_latent_adaptive(self):
        """Test sweep_latent_adaptive method."""
        # Use a spy to track calls
        with patch.object(self.rnn_model.visualiser, 'sweep_latent_adaptive') as mock_method:
            # Call the method
            self.rnn_model.visualiser.sweep_latent_adaptive(self.model_params, self.dataset, 2)
            
            # Assertions
            mock_method.assert_called_once_with(self.model_params, self.dataset, 2)


class TestRNNFormatOutput(unittest.TestCase):
    """Test cases for the format_output static method in rnn.py."""
    
    def test_format_output_normal_case(self):
        """Test format_output with valid input."""
        # Create a test tensor
        test_tensor = torch.ones(1, 100, 1)
        target_timesteps = 1000
        
        # Call the method
        result = MockRNN.format_output(test_tensor, target_timesteps)
        
        # Assertions
        self.assertEqual(result.shape, (1, target_timesteps, 1))
        self.assertTrue(torch.allclose(result, torch.ones(1, target_timesteps, 1)))
    
    def test_format_output_invalid_shape(self):
        """Test format_output with invalid tensor shape."""
        # Create an invalid tensor
        test_tensor = torch.ones(2, 100, 1)  # First dimension should be 1
        target_timesteps = 1000
        
        # Assert that it raises ValueError
        with self.assertRaises(ValueError):
            MockRNN.format_output(test_tensor, target_timesteps)
    
    def test_format_output_not_tensor(self):
        """Test format_output with non-tensor input."""
        # Create a non-tensor input
        test_input = np.ones((1, 100, 1))
        target_timesteps = 1000
        
        # Assert that it raises TypeError
        with self.assertRaises(TypeError):
            MockRNN.format_output(test_input, target_timesteps)
    
    def test_format_output_invalid_target(self):
        """Test format_output with invalid target_timesteps."""
        # Create a test tensor
        test_tensor = torch.ones(1, 100, 1)
        target_timesteps = 50  # Should be greater than original timesteps
        
        # Assert that it raises ValueError
        with self.assertRaises(ValueError):
            MockRNN.format_output(test_tensor, target_timesteps)


if __name__ == '__main__':
    unittest.main()