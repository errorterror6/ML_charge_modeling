import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch, random, glob, os
from torch.utils.data import DataLoader, TensorDataset
import visualisation
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import matplotlib.colors as colors
import matplotlib.cm as cmx

from . import encoders
from . import decoders

from abc import ABC, abstractmethod

import sys
sys.path.append('..')
import parameters
import loader




class VAE(nn.Module):
    def __init__(self, enc, dec, loss_function=None):
        super(VAE, self).__init__()
        self.encoder = enc
        self.decoder = dec
        self.loss_fn = torch.nn.MSELoss() if loss_function is None else loss_function
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=parameters.model_params['lr']
        )
        self.visualiser = self.Visualiser(self)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, x):
        """
        Input x should be whole sequence 
        of shape [batch_size, sequence_length, feature_dim]
        expected [x, 70, 6].
        
        
        """
        mu, log_var = self.encoder.forward(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, log_var, z
    
    def loss_function(self, traj, reconstruction, mu, log_var, beta=1.0):
        # During training, use all dimensions for reconstruction loss
        # Both traj and reconstruction should be [batch_size, seq_len, 6]
        # TODO: revert loss to all 6 dimensions, remove the 0.1 multiplier on beta.
        recon_loss = self.loss_fn(reconstruction[:, :, 0:2], traj[:, :, 0:2])
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # return recon_loss, recon_loss, KLD
        # print(f"recon loss: {recon_loss}")
        return recon_loss + 0.1* beta * KLD, recon_loss, KLD
    
    
    
    

    def train_step(self, input_data):
        #input data is expected to be [16, 70, 6]
        self.train()
        reversed_data = loader.reverse_traj(input_data)

        reconstruction, mu, log_var, z = self.forward(reversed_data)
        loss, recon_loss, kl_loss = self.loss_function(input_data, reconstruction, mu, log_var)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        return loss, recon_loss, kl_loss
    
    def infer_step(self, input_data, printout=False):
        """
        Perform inference step from a trajectory.
        
        Args:
            input_data (Tensor): Input trajectory [batch_size, seq_len, obs_dim]
            output_time_points (Tensor): Time points to predict at
            
        Returns:
            tuple: (pred_x, pred_z)
                - pred_x: Predicted observations [batch_size, time_len, obs_dim]
                - pred_z: Predicted latent states [batch_size, time_len, latent_dim]
        """
        # Inference mode (no gradient computation)
        self.eval()
        with torch.no_grad():
            # Process input data
            reversed_data = loader.reverse_traj(input_data)
            
            # Run model inference
            
            reconstruction, mu, log_var, z = self.forward(reversed_data)  
            if printout:
                print("input data:", reversed_data)   
                print("reconstruction:", reconstruction) 
            # print("latent z:", z)
        return reconstruction, z, self.eval_loss_fn(input_data, reconstruction)
        
    
    def eval_loss_fn(self, traj, reconstruction):
        # traj is expected to be [16, 70, 6], same as reconstruction
        #with features [x, t, y, intensity, bias, delay]
        
        # Only evaluate on first dimension (x) for consistent evaluation metric
        traj = traj[:, :, 0:1]  # Take only the first feature dimension
        reconstruction = reconstruction[:, :, 0:1]  # Take only the first feature dimension
        recon_loss = self.loss_fn(reconstruction, traj)
        return recon_loss
    
    def eval_step(self, input_data):
        self.eval()
        
        with torch.no_grad():
            reversed_data = loader.reverse_traj(input_data)
            reconstruction, mu, log_var, z = self.forward(reversed_data)
            total_loss, feat_loss, kl_loss = self.loss_function(input_data, reconstruction, mu, log_var)
            eval_loss = self.eval_loss_fn(input_data, reconstruction) 
                  
        return eval_loss, (total_loss, feat_loss, kl_loss)
             
    
    def train_nepochs(self, n_epochs, m=parameters.model_params, v=parameters.vae_params, r=parameters.records):
        train, val, orig = loader.get_formatted_data()
        # train = train.to(m['device'])
        # val = val.to(m['device'])
        # orig = orig.to(m['device'])
        total_loss_history = []
        recon_loss_history = []
        kl_loss_history = []
        eval_loss_history = []
        
        print(f"logs: vae: train_nepochs: training for {n_epochs} epochs.")
        for epoch in range(n_epochs):
            total_loss = 0
            recon_loss = 0
            kl_loss = 0
            eval_loss = 0
            
            try:
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                
                #send each bath to training.
                for batch_idx, (x, y, meta) in enumerate(train):
                    input_data = loader.compile_stacked_data(x, y, meta)
                    self.train_step(input_data)
                    
                #TODO: eval each batch and get the loss. record.
                for batch_idx, (x, y, meta) in enumerate(val):
                    input_data = loader.compile_stacked_data(x, y, meta)
                    eval_loss, loss = self.eval_step(input_data)
                    total_loss += loss[0]
                    recon_loss += loss[1]
                    kl_loss += loss[2]
                    eval_loss += eval_loss
                    
                total_loss /= len(val)
                recon_loss /= len(val)
                kl_loss /= len(val)
                eval_loss /= len(val)
                total_loss_history.append(total_loss)
                recon_loss_history.append(recon_loss)
                kl_loss_history.append(kl_loss)
                eval_loss_history.append(eval_loss)
                
                print(f"Epoch {epoch}: Eval Loss {eval_loss:.3f}, "
                    f"KL Loss {kl_loss:.2f}, Feature Loss {recon_loss:.2f}, Total Loss {total_loss:.2f}")
                
                    
            except KeyboardInterrupt:
                print("logs: vae: train_nepochs: Training interrupted by keyboard.")
                exit(1)
            
            #do a final evaluation over 150 datapoints
            
        return n_epochs, eval_loss_history, (total_loss_history, recon_loss_history, kl_loss_history)

    def format_output(self, pred_x, target_timesteps=1000):
        """             
        Extrapolates the pred_x tensor to a specified target length using linear interpolation
        between known time points.
        
        Args:
            pred_x: A PyTorch tensor of shape torch.Size([1, initial_timesteps, 1]).
            target_timesteps: The desired number of timesteps in the extrapolated tensor (e.g., 1000).
                            Must be greater than the initial number of timesteps in pred_x.
        Returns:
            pred_x_extrapolated: A PyTorch tensor of shape torch.Size([1, target_timesteps, 1]) with linearly interpolated values.
        """
        if not isinstance(pred_x, torch.Tensor):
            raise TypeError("Input pred_x must be a PyTorch tensor.")
        if len(pred_x.shape) != 3 or pred_x.shape[0] != 1 or pred_x.shape[2] != 1:
            raise ValueError(f"Input pred_x must have shape torch.Size([1, initial_timesteps, 1]), but got {pred_x.shape}")
        if target_timesteps <= pred_x.shape[1]:
            raise ValueError("target_timesteps must be greater than the original timesteps for extrapolation.")
        
        original_timesteps = pred_x.shape[1]
        pred_x_extrapolated = torch.zeros(1, target_timesteps, 1)  # Initialize with zeros
        
        # Calculate the scaling factor between original and target timesteps
        scale_factor = original_timesteps / target_timesteps
        
        for j in range(target_timesteps):
            # Calculate the exact position in the original time scale
            original_pos = j * scale_factor
            
            # Get the indices of the two nearest points in the original sequence
            lower_idx = int(original_pos)
            upper_idx = min(lower_idx + 1, original_timesteps - 1)
            
            # If we're exactly on a known point or beyond the range, no interpolation needed
            if lower_idx >= original_timesteps - 1:
                pred_x_extrapolated[:, j, :] = pred_x[:, original_timesteps - 1, :]
            elif lower_idx == original_pos:
                pred_x_extrapolated[:, j, :] = pred_x[:, lower_idx, :]
            else:
                # Calculate interpolation weight (how far we are between the two known points)
                weight = original_pos - lower_idx
                
                # Linear interpolation: value = (1-w) * v1 + w * v2
                pred_x_extrapolated[:, j, :] = (1 - weight) * pred_x[:, lower_idx, :] + weight * pred_x[:, upper_idx, :]
        
        return pred_x_extrapolated
        
    def save_model(self, model_params):
        pass
    
    @classmethod
    def create(cls, m=parameters.model_params, v=parameters.vae_params):
        """
        Create VAE model with encoder and decoder.
        
        Args:
            m (dict): Model parameters
            v (dict): VAE parameters
        
        Returns:
            VAE: Instance of VAE model
        """
        if parameters.trainer == 'RNN-VAE':
            encoder = encoders.RNNEncoder(m, v).to(m['device'])
            decoder = decoders.RNNDecoder(m, v).to(m['device'])
        elif parameters.trainer == 'MLP-VAE':
            encoder = encoders.MLPEncoder(m, v).to(m['device'])
            decoder = decoders.MLPDecoder(m, v).to(m['device'])
        elif parameters.trainer == 'LSTM-VAE':
            encoder = encoders.LSTMEncoder(m, v).to(m['device'])
            decoder = decoders.LSTMDecoder(m, v).to(m['device'])
        else:
            print(f"auto: vae: Error: Unknown trainer type '{parameters.trainer}'.")
            exit(1)
        
        return cls(encoder, decoder)
    
    class Visualiser:
        def __init__(self, model):
            self.model = model
            pass
            
        def plot_training_loss(self, model_params=parameters.model_params, save=False, split=False, plot_total=True, plot_MSE=True, plot_KL=True):
            # NOTE: uncomment or comment these to toggle between.
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=False, plot_KL=False)
            # visualisation.plot_training_loss(model_params, save=save, split=True, plot_total=False, plot_MSE=plot_MSE, plot_KL=plot_KL)

        def display_random_fit(self, model_params=parameters.model_params, dataset=parameters.dataset, show=False, save=True, random_samples=True, test=True):
            """
            Display model fit on random samples from the dataset.
            
            This function selects random trajectory samples, runs model inference on them,
            and plots both the original data and model predictions for visual comparison.
            
            Parameters
            ----------
            model_params : dict
                Dictionary containing model parameters and components
                Required keys: 'func', 'rec', 'dec', 'optim', 'device', 'epochs', 'lr', 'beta', 'folder'
            dataset : dict
                Dictionary containing dataset components
                Required keys: 'trajs', 'times', 'y'
            show : bool, default=True
                Whether to display the plot interactively
            save : bool, default=False
                Whether to save the plot to disk
            random_samples : bool, default=True
                If True, randomly sample trajectories; otherwise use sequential samples
            
            Returns
            -------
            None
                The function creates and optionally saves a plot but doesn't return any values
            """
            # Extract data from dataset - use original data without missing points
            trajectories = dataset['trajs']  # Original data (with no missing points)
            metadata = dataset['y']          # Contains parameters like intensity, bias, delay
            
            train_trajs = dataset['train_trajs']  # Trajectories with missing data points
            train_times = dataset['train_times']  # Time points with missing data points
            trajs = dataset['trajs']
            times = dataset['times']

            # print(f"Debug: B-VAE: visualisation: display_random: trajectories shape: {trajectories.shape}")
            # print(f"Debug: B-VAE: visualisation: display_random: time_points shape: {time_points.shape}")

            # Extract model components
            model_func = model_params['func']
            encoder = model_params['rec']
            decoder = model_params['dec']
            optimizer = model_params['optim']
            device = model_params['device']
            epoch_num = model_params['epochs']
            epoch = model_params['epochs']
            
            # Set up figure with subplots (one per trajectory dimension)
            num_dims = trajectories[0].shape[-1]
            # print("num_dims:", num_dims)
            fig_width = 7
            fig_height = 4 * num_dims
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # Create a subplot for each dimension
            axes = [fig.add_subplot(num_dims, 1, i+1) for i in range(num_dims)]

            # Select indices of trajectories to plot
            sample_indices = list(range(len(trajectories)))
            if random_samples:
                random.shuffle(sample_indices)
                # Downsample to avoid cluttering the plot
                sample_indices = sample_indices[::30]
            else:
                sample_indices = model_params['plot']

            # Create colormap for different trajectories
            color_norm = colors.Normalize(vmin=0, vmax=len(sample_indices))
            color_map = cmx.ScalarMappable(norm=color_norm, cmap='brg')
            
            # Iterate over selected trajectories
            loss_list = []
            for idx, traj_idx in enumerate(sample_indices):
                # Get color for this trajectory
                color = color_map.to_rgba(idx)
                
                # Prepare input trajectory tensor
                traj_tensor = train_trajs[traj_idx].to(device)
                time_tensor = train_times[traj_idx].to(device)
                meta_tensor = metadata[traj_idx].to(device)
                
                input_tensor = loader.compile_stacked_data(traj_tensor, time_tensor, meta_tensor)
                
                # Create time points for prediction (denser than original data)
                pred_times = np.linspace(0, 2.5, 1000) + 1  # +1 accounts for time bias
                time_1k_tensor = torch.Tensor(np.linspace(0,2.5,1000)).to(device)
                
                # Run model inference
                pred_x, pred_z, loss = self.model.infer_step(input_tensor, printout=True)
                loss_list.append(loss.detach().cpu().numpy())
                if test:
                    pred_x = loader.interpolate_traj(pred_x, time_1k_tensor)
                else:
                    pred_x = self.model.format_output(pred_x[:,:,0:1])
                pred_x = pred_x.detach().cpu().numpy()[0]
                _traj = trajs[traj_idx].detach().cpu()
                _t = times[traj_idx].detach().cpu()

                for l in range(num_dims):
                    u = 0
                    
                    # ax[k].set_ylim(-.8, .8)
                    sc_ = 50*1e2/1e3
                    
                    # plot original and predicted trajectories
                    axes[l].plot(_t, _traj[:, l+u]/sc_, '.', alpha = 0.6, color = color)
                    axes[l].plot(pred_times - 1.0, pred_x[:, l+u]/sc_, '-', label = '{:.1f} J$, {:.1f} V, {:.0e} s'.format(metadata[traj_idx][0], metadata[traj_idx][1], metadata[traj_idx][2]),
                            linewidth = 2, alpha = 0.4, color = color)

            mean_loss = np.mean(loss_list)
            plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
            plt.ylabel('Charge [mA]')
            # tile includes epoch number, learning rate atnd beta
            plt.title('Epoch: {}, lr: {:.1e}, loss: {:.1e}'.format(epoch, model_params['lr'], mean_loss))

            # plt.xscale('log')
            plt.legend(loc='upper right', title='Intensity, Bias, Delay')

            plt.tight_layout()

            if show:
                plt.show()
            if save:
                # save as a png]
                # todo: change
                folder = model_params['folder']
                # if saves folder does not exist create it
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig(f"{folder}/training_epoch_{epoch:04d}.png", dpi=300)
                
            plt.close('all')

                
                
            return
                
            #     # Extract trajectory and combine with time for interpolation
            #     batch_size, seq_len, obs_dim = pred_x.shape
                
            #     # We still focus on the first dimension (x) for visualization
            #     trajectory_values = pred_x[:, :, 0]  # Get first dimension of reconstruction
            #     # Stack along the last dimension to match the expected format for interpolate_trajectory
            #     # Shape [batch_size, seq_len, 2] where each point has (traj_value, time_value)
            #     interpolation_input = torch.stack([trajectory_values.squeeze().unsqueeze(dim=1), time_tensor], dim=-1)
            #     # print(f"interpolation_input:", interpolation_input)
            #     # Convert tensors to numpy arrays before interpolation
            #     interpolation_input_np = interpolation_input.detach().cpu().numpy()
            #     time_1k_tensor_np = time_1k_tensor.detach().cpu().numpy()
                
                
                
            #     # Run interpolation
            #     interpolated_result_np = loader.interpolate_trajectory(interpolation_input_np, time_1k_tensor_np)
            #     # print(f"interpolation output:", interpolation_input_np)
            #     # Convert back to tensor and extract first dimension
            #     interpolated_result = torch.tensor(interpolated_result_np, device=device)[:, :, 0]
            #     pred_x = interpolated_result
                
            #     #Run model inference for loss reasons only
            #     loss_x, _ = self.model.eval_step(input_tensor)
                
            #     loss_list.append(loss_x.item())
            #     # print(f"Debug: B-VAE: visualisation: display_random: pred_x shape: {pred_x.shape}")

            #     # Convert prediction to numpy for plotting
            #     # print(f"pred_x shape: {pred_x.shape}")
            #     # pred_x shape here is [1, 1000]
            #     pred_x_np = pred_x.detach().cpu().numpy()
            #     # print(f"pred_x_np shape before indexing: {pred_x_np.shape}")
                
            #     # Remove batch dimension if it exists
            #     if len(pred_x_np.shape) > 1 and pred_x_np.shape[0] == 1:
            #         pred_x_np = pred_x_np.reshape(pred_x_np.shape[1])
                
            #     # print(f"pred_x_np shape after reshaping: {pred_x_np.shape}")
                
            #     # Get original trajectory and time data
            #     orig_traj = trajectories[traj_idx].detach().cpu()
            #     orig_time = time_points[traj_idx].detach().cpu()
            #     # print(f"orig_traj shape: {orig_traj.shape}")
            #     # Scaling factor for better visualization
            #     scale_factor = 50 * 1e2 / 1e3
                
            #     # Plot each dimension
            #     for dim in range(num_dims):
            #         # Plot original trajectory points
            #         axes[dim].plot(orig_time, orig_traj[:, dim] / scale_factor, 
            #                 '.', alpha=0.6, color=color)
                    
            #         # For the model prediction, only use pred_x_np as a 1D array for all dimensions
            #         # This is a temporary fix - all dimensions show the same prediction
            #         axes[dim].plot(time_1k_tensor_np - 1.0, pred_x_np / scale_factor, 
            #                 '-', linewidth=2, alpha=0.4, color=color,
            #                 label='{:.1f} J$, {:.1f} V, {:.0e} s'.format(
            #                     metadata[traj_idx][0], metadata[traj_idx][1], metadata[traj_idx][2]))
                    
            # # Add labels and title
            # plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
            # plt.ylabel('Charge [mA]')
            # plt.title('Epoch: {}, lr: {:.1e}, beta: {:.1e}, loss: {:.1e}'.format(
            #     epoch_num, model_params['lr'], model_params['beta'], np.mean([l for l in loss_list])))

            # # Add legend
            # plt.legend(loc='upper right', title='Intensity, Bias, Delay')
            # plt.tight_layout()

            # # Show or save the figure
            # if show:
            #     plt.show()
            # if save:
            #     save_dir = model_params['folder']
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     fig.savefig(f"{save_dir}/training_epoch_{epoch_num:04d}.png", dpi=300)
