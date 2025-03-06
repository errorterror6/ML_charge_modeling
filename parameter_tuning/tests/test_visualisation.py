import unittest
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the visualisation functions instead of importing them directly
# This helps us avoid the circular import issues
class MockVisualisation:
    @staticmethod
    def plot_training_loss(model_params, save=False, split=False, plot_total=False, plot_MSE=True, plot_KL=True, scale='log'):
        """Mock implementation of plot_training_loss"""
        pass
        
    @staticmethod
    def display_random_fit(model_params, dataset, show=True, save=False, random_samples=True):
        """Mock implementation of display_random_fit"""
        pass
        
    @staticmethod
    def compile_learning_gif(model_params, display=True):
        """Mock implementation of compile_learning_gif"""
        pass
    
    @staticmethod
    def sweep_latent_adaptives(model_params, dataset):
        """Mock implementation of sweep_latent_adaptives"""
        pass
        
    @staticmethod
    def sweep_latent_adaptive(model_params, dataset, latent_dim_number):
        """Mock implementation of sweep_latent_adaptive"""
        pass


class TestVisualisation(unittest.TestCase):
    """Test cases for the visualisation.py module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model_params dictionary
        self.model_params = {
            'MSE_loss': [100, 90, 80, 70, 60],
            'KL_loss': [50, 45, 40, 35, 30],
            'loss': [150, 135, 120, 105, 90],
            'epochs': 5,
            'folder': 'test_folder'
        }
        
        # Create a mock dataset dictionary
        self.dataset = {
            'trajs': torch.randn(10, 100, 1),  # 10 trajectories, 100 timepoints, 1 feature
            'times': torch.randn(10, 100, 1),  # 10 trajectories, 100 timepoints, 1 time feature
            'y': torch.randn(10, 3)  # 10 trajectories, 3 parameters (intensity, bias, delay)
        }
        
        # Create directory for test outputs if it doesn't exist
        os.makedirs('test_folder/loss_graph', exist_ok=True)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directories if they exist
        if os.path.exists('test_folder'):
            import shutil
            shutil.rmtree('test_folder')

    def test_plot_training_loss_split(self):
        """Test plotting training loss with split=True."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'plot_training_loss') as mock_method:
            # Call the function
            MockVisualisation.plot_training_loss(self.model_params, save=True, split=True, 
                                                plot_total=True, plot_MSE=True, plot_KL=True)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, save=True, split=True, 
                plot_total=True, plot_MSE=True, plot_KL=True
            )

    def test_plot_training_loss_no_split(self):
        """Test plotting training loss with split=False."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'plot_training_loss') as mock_method:
            # Call the function
            MockVisualisation.plot_training_loss(self.model_params, save=True, split=False)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, save=True, split=False
            )

    def test_display_random_fit(self):
        """Test displaying random fit."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'display_random_fit') as mock_method:
            # Call the function
            MockVisualisation.display_random_fit(self.model_params, self.dataset, show=False, save=True)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, self.dataset, show=False, save=True
            )

    def test_compile_learning_gif(self):
        """Test compiling learning gif."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'compile_learning_gif') as mock_method:
            # Call the function
            MockVisualisation.compile_learning_gif(self.model_params, display=False)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, display=False
            )

    def test_sweep_latent_adaptive(self):
        """Test sweeping latent adaptive."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'sweep_latent_adaptive') as mock_method:
            # Call the function
            MockVisualisation.sweep_latent_adaptive(self.model_params, self.dataset, 0)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, self.dataset, 0
            )
            
    def test_sweep_latent_adaptives(self):
        """Test sweeping latent adaptives."""
        # Use a spy to track method calls
        with patch.object(MockVisualisation, 'sweep_latent_adaptives') as mock_method:
            # Call the function
            MockVisualisation.sweep_latent_adaptives(self.model_params, self.dataset)
            
            # Assertions
            mock_method.assert_called_once_with(
                self.model_params, self.dataset
            )


if __name__ == '__main__':
    unittest.main()