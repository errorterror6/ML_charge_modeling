"""
Inference Module for Stochastic Hidden Jump Neural Network (SHJNN)

This module provides functions for running inference with the trained SHJNN model:
1. Inference step function factory (two modes: latent input or trajectory input)
2. ODE-based trajectory prediction in the latent space
3. Decoding predicted latent trajectories to observation space
"""

import torch
from torchdiffeq import odeint


def make_infer_step(dynamics_func, recognition_network, decoder, optimizer, device, input_mode='traj', sample=True):
    """
    Create an inference step function.
    
    This factory function returns a function that performs prediction:
    either from a latent state directly or by first encoding a trajectory.
    
    Args:
        dynamics_func (nn.Module): ODE function for latent dynamics
        recognition_network (nn.Module): Recognition network for encoding trajectories
        decoder (nn.Module): Decoder network for decoding latent states
        optimizer (torch.optim.Optimizer): Optimizer (not used during inference)
        device (torch.device): Device to run computations on
        input_mode (str): Input mode - 'latent' or 'traj'
        sample (bool): Whether to sample from latent distribution or use mean
        
    Returns:
        function: Inference step function
    """
    
    # Inference function for direct latent input
    if input_mode == 'latent':
        def infer_step(z0, time_points):
            """
            Perform inference step from a latent state.
            
            Args:
                z0 (Tensor): Initial latent state [batch_size, latent_dim]
                time_points (Tensor): Time points to predict at
                
            Returns:
                tuple: (pred_x, pred_z)
                    - pred_x: Predicted observations [batch_size, time_len, obs_dim]
                    - pred_z: Predicted latent states [batch_size, time_len, latent_dim]
            """
            # Inference mode (no gradient computation)
            with torch.no_grad():
                # Set models to evaluation mode
                recognition_network.eval()
                dynamics_func.eval()
                decoder.eval()
                
                # Predict latent trajectory by solving ODE
                pred_z = odeint(
                    dynamics_func, 
                    z0, 
                    time_points.squeeze()
                ).permute(1, 0, 2)
                
                # Decode latent trajectory to observation space
                pred_x = decoder(pred_z)
                
            return pred_x, pred_z
    
    # Inference function for trajectory input
    else:
        def infer_step(trajectory, time_points):
            """
            Perform inference step from a trajectory.
            
            Args:
                trajectory (Tensor): Input trajectory [batch_size, seq_len, obs_dim]
                time_points (Tensor): Time points to predict at
                
            Returns:
                tuple: (pred_x, pred_z)
                    - pred_x: Predicted observations [batch_size, time_len, obs_dim]
                    - pred_z: Predicted latent states [batch_size, time_len, latent_dim]
            """
            # Inference mode (no gradient computation)
            with torch.no_grad():
                # Set models to evaluation mode
                recognition_network.eval()
                dynamics_func.eval()
                decoder.eval()
                
                # --- Encode trajectory (in reverse time) ---
                
                # Initialize recognition network hidden state
                hidden_state = recognition_network.initHidden().to(device)[:1, :]
                
                # Process trajectory in reverse order
                for t in reversed(range(trajectory.size(1))):
                    # Get trajectory sample at time t
                    observation = trajectory[:, t, :]
                    
                    # Process through recognition network
                    output, hidden_state = recognition_network.forward(observation, hidden_state)
                
                # --- Infer initial latent state ---
                
                # Split recognition output into mean and log-variance
                latent_dim = output.size()[1] // 2
                latent_mean, latent_logvar = output[:, :latent_dim], output[:, latent_dim:]
                
                # Either sample from the distribution or use the mean
                if sample:
                    # Sample from posterior using reparameterization trick
                    epsilon = torch.randn(latent_mean.size()).to(device)
                    z0 = epsilon * torch.exp(0.5 * latent_logvar) + latent_mean
                else:
                    # Use mean directly
                    z0 = latent_mean
                
                # --- Predict trajectory ---
                
                # Predict latent trajectory by solving ODE
                pred_z = odeint(
                    dynamics_func, 
                    z0, 
                    time_points.squeeze()
                ).permute(1, 0, 2)
                
                # Decode latent trajectory to observation space
                pred_x = decoder(pred_z)
                
            return pred_x, pred_z
    
    # Return the appropriate inference function
    return infer_step