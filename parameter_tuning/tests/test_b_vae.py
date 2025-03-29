import unittest
import sys
import os
import torch
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import shjnn directly
from libs import shjnn

# Create a minimal B_VAE-like class for testing just the batch evaluation
class MockBVAE:
    """A simplified version of B_VAE class for testing batch processing."""
    
    def __init__(self):
        pass
        
    def eval(self, model_params, dataset):
        """Simplified version of b_vae.py's eval function that only tests batch processing"""
        # Extract data from dataset 
        trajectories = dataset['trajs']
        time_points = dataset['times']
        
        # Extract model components
        model_func = model_params['func']
        encoder = model_params['rec'] 
        decoder = model_params['dec']
        optimizer = model_params['optim']
        device = model_params['device']
        
        # Create inference step function
        infer_step = shjnn.make_infer_step(
            model_func, encoder, decoder, optimizer, device,
            input_mode='traj', sample=False
        )
        
        # Begin evaluation
        total_samples = len(trajectories)
        print(f"logs: test_b_vae: eval: evaluating over {total_samples} samples.")
        loss_list = []
        
        try:
            # Process in batches for better GPU utilization
            batch_size = model_params['n_batch']
            n_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_idx in range(n_batches):
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Determine batch range
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = list(range(start_idx, end_idx))
                batch_size_actual = len(batch_indices)
                
                # Skip empty batches
                if batch_size_actual == 0:
                    continue
                    
                # Prepare batch tensors
                batch_trajs = []
                batch_times = []
                
                for traj_idx in batch_indices:
                    batch_trajs.append(trajectories[traj_idx])
                    batch_times.append(time_points[traj_idx])
                
                # Stack into batch tensors and move to device
                traj_tensor = torch.stack(batch_trajs).to(device)
                time_tensor = torch.stack(batch_times).to(device)
                
                # We need to process each sample in the batch separately
                # because infer_step's ODE solver expects a specific format for time_points
                batch_pred_x = []
                for i in range(batch_size_actual):
                    # Process individual samples from the batch
                    sample_traj = traj_tensor[i:i+1]
                    sample_time = time_tensor[i:i+1]
                    
                    # Run inference step
                    sample_pred_x, _ = infer_step(sample_traj, sample_time)
                    batch_pred_x.append(sample_pred_x)
                
                # Combine results back into a batch
                pred_x = torch.cat(batch_pred_x, dim=0)
                
                # Compute individual losses for each sample
                for i in range(batch_size_actual):
                    # Extract individual predictions and targets
                    pred_x_i = pred_x[i:i+1]
                    traj_tensor_i = traj_tensor[i:i+1]
                    
                    # Compute individual sample loss
                    individual_loss = torch.nn.MSELoss()(pred_x_i.to(device), traj_tensor_i.to(device))
                    loss_list.append(individual_loss.item())
                
                # Calculate and report batch average loss
                batch_avg_loss = sum(loss_list[-batch_size_actual:]) / batch_size_actual
                
                # Debug output
                print(f"logs: test_b_vae: eval: batch {batch_idx+1}/{n_batches}, avg loss: {batch_avg_loss:.6f}")
            
            # Return mean loss
            if loss_list:
                mean_loss = np.mean(loss_list)
                print(f"logs: test_b_vae: eval: mean loss over {total_samples} samples: {mean_loss:.6f}")
                return mean_loss
            else:
                return float('inf')
            
        except Exception as e:
            print(f"Error during batch evaluation: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')

class TestBVAE(unittest.TestCase):
    """Test the B-VAE class and its evaluation with batch processing."""
    
    def setUp(self):
        """Set up test data and model components."""
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create small dataset for testing
        self.num_samples = 30  # Small number for fast tests
        self.seq_len = 10
        self.feature_dim = 1
        self.batch_size = 8
        
        # Model dimensions
        self.latent_dim = 4
        self.nhidden = 32
        self.rnn_nhidden = 32
        
        # Create mock dataset
        self.dataset = self._create_mock_dataset()
        
        # Create model parameters
        self.model_params = self._setup_model_params()
        
        # Create the MockBVAE instance
        self.b_vae = MockBVAE()
    
    def _create_mock_dataset(self):
        """Create a mock dataset for testing."""
        # Create random trajectories and times
        trajs = [torch.randn(self.seq_len, self.feature_dim) for _ in range(self.num_samples)]
        times = [torch.linspace(0, 1, self.seq_len).unsqueeze(-1) for _ in range(self.num_samples)]
        
        # Create mock metadata
        y = np.random.randn(self.num_samples, 4)  # intensity, voltage, delay, thickness
        
        # Create training versions (duplicates for this test)
        train_trajs = [t.clone() for t in trajs]
        train_times = [t.clone() for t in times]
        
        # Assemble dataset
        dataset = {
            'trajs': trajs,
            'times': times,
            'y': y,
            'train_trajs': train_trajs,
            'train_times': train_times,
            'missing_idx': None,
            'drop_number': 0
        }
        return dataset
    
    def _setup_model_params(self):
        """Set up model parameters for testing."""
        # Initialize model components
        func, rec, dec, optim, _ = shjnn.init_model(
            self.latent_dim, 
            self.nhidden, 
            self.rnn_nhidden, 
            self.feature_dim,
            self.batch_size,
            1e-3,
            device=self.device
        )
        
        # Set up model params dictionary
        model_params = {
            'nhidden': self.nhidden,
            'rnn_nhidden': self.rnn_nhidden,
            'obs_dim': self.feature_dim,
            'latent_dim': self.latent_dim,
            'lr': 1e-3,
            'n_batch': self.batch_size,
            'beta': 0.1,
            'device': self.device,
            'func': func,
            'rec': rec, 
            'dec': dec,
            'optim': optim,
            'epochs': 0,
            'loss': [],
            'MSE_loss': [],
            'KL_loss': []
        }
        
        return model_params
    
    def test_batch_evaluation(self):
        """Test that batch evaluation runs without errors."""
        # Run the evaluation
        try:
            loss = self.b_vae.eval(self.model_params, self.dataset)
            
            # Check that loss is a valid number
            self.assertTrue(isinstance(loss, float))
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))
            
            print(f"B-VAE batch evaluation successful with loss: {loss:.6f}")
        except Exception as e:
            self.fail(f"B-VAE batch evaluation raised an exception: {e}")
    
if __name__ == '__main__':
    unittest.main()