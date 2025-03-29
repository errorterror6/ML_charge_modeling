"""
Training Module for Stochastic Hidden Jump Neural Network (SHJNN)

This module provides functions for training the SHJNN model:
1. Training step function factory
2. Log-normal probability density function
3. KL divergence calculation
"""

import numpy as np
import torch
from torchdiffeq import odeint


def make_train_step(dynamics_func, recognition_network, decoder, optimizer, device, noise_std=0.3, beta=1.0):
    """
    Create a training step function.
    
    This factory function returns a function that performs a complete
    training step: forward pass, loss calculation, and parameter update.
    
    Args:
        dynamics_func (nn.Module): ODE function for latent dynamics
        recognition_network (nn.Module): Recognition network for encoding trajectories
        decoder (nn.Module): Decoder network for decoding latent states
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        device (torch.device): Device to run computations on
        noise_std (float): Standard deviation of observation noise
        beta (float): KL divergence weight in the loss function
        
    Returns:
        function: Training step function that takes trajectory and time inputs
    """
    
    def train_step(trajectory, time_points):
        """
        Perform a single training step.
        
        Args:
            trajectory (Tensor): Trajectory data [batch_size, seq_length, feature_dim]
            time_points (Tensor): Time points [batch_size, seq_length, 1]
            
        Returns:
            tuple: (total_loss, reconstruction_loss, kl_loss)
                - total_loss: Combined loss value (scalar)
                - reconstruction_loss: Negative log likelihood term (scalar)
                - kl_loss: KL divergence term (scalar)
        """
        # Set models to training mode
        recognition_network.train()
        dynamics_func.train()
        decoder.train()
        
        # --- Encode trajectory (in reverse time) ---
        
        # Initialize recognition network hidden state
        hidden_state = recognition_network.initHidden().to(device)
        
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
        
        # Sample from posterior using reparameterization trick
        epsilon = torch.randn(latent_mean.size()).to(device)
        latent_z0 = epsilon * torch.exp(0.5 * latent_logvar) + latent_mean
        
        # --- Compute state predictions ---
        
        # Set ODE solver absolute tolerance
        atol = 1e-6
        
        # Solve ODE to get latent state trajectory
        pred_z = odeint(
            dynamics_func,
            latent_z0,
            time_points[0, ...].squeeze(),
            atol=atol
        ).permute(1, 0, 2)
        
        # Decode latent trajectory to observation space
        pred_x = decoder(pred_z)
        
        # --- Compute loss ---
        
        # Observation noise model
        noise_std_tensor = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2.0 * torch.log(noise_std_tensor).to(device)
        
        # Log probability of observations under predicted distribution (reconstruction loss)
        log_px = log_normal_pdf(trajectory, pred_x, noise_logvar).squeeze().mean(-1)
        log_px = log_px / trajectory.size(0)  # Normalize by batch size
        
        # Prior distribution parameters (standard normal)
        prior_mean = prior_logvar = torch.zeros(latent_z0.size()).to(device)
        
        # KL divergence between posterior and prior
        kl_divergence = normal_kl(latent_mean, latent_logvar, prior_mean, prior_logvar).mean(-1)
        
        # Total loss (negative ELBO)
        loss = torch.mean(-log_px + beta * kl_divergence)
        
        # --- Parameter update ---
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Return loss components
        return loss.item(), torch.mean(log_px).detach().cpu().numpy(), torch.mean(kl_divergence).detach().cpu().numpy()
    
    # Return the training step function
    return train_step


def log_normal_pdf(x, mean, logvar):
    """
    Compute log probability density of a normal distribution.
    
    Args:
        x (Tensor): Observation values
        mean (Tensor): Mean of the normal distribution
        logvar (Tensor): Log-variance of the normal distribution
        
    Returns:
        Tensor: Log probability density values
    """
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    """
    Compute KL divergence between two normal distributions.
    
    Args:
        mu1 (Tensor): Mean of the first distribution
        lv1 (Tensor): Log-variance of the first distribution
        mu2 (Tensor): Mean of the second distribution
        lv2 (Tensor): Log-variance of the second distribution
        
    Returns:
        Tensor: KL divergence values
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0
    
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    
    return kl