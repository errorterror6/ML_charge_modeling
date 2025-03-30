#!/usr/bin/env python3

import sys
sys.path.append('..')
import parameters
from torch import nn
import torch, random, glob, os
from torch.utils.data import DataLoader, TensorDataset
import visualisation
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import matplotlib.colors as colors
import matplotlib.cm as cmx

import loader

import sys
sys.path.append('../../libs/')
import shjnn

class RNN(nn.Module):

    def __init__(self, m=parameters.model_params):
        super().__init__()
        
        self.model_params = m
        self.criterion = nn.MSELoss()
        # input : (y_t, time_t) to hidden
        # NOTE: currently the RNN takes in 2 features, the charge and the time, but only the charge is used for prediction and time is
        # simply discarded. This was built this way to allow future expandability to incorporate time - to do this, go to 
        # forward_step and change the loss function to include the time feature.
        self.temporal = nn.RNN(
            input_size=2,
            hidden_size=m['rnn_nhidden'],
            # use relu + clip_gradient if poor results with tanh
            nonlinearity='tanh',
            device=m['device'],
            batch_first=True
            )
        # TODO: encode experimental variables into the hidden layer as init.
        # hidden to output (y_t+1, time_t+1)
        self.h2h0 = nn.Linear(m['rnn_nhidden'], m['rnn_linear1']).to(m['device'])
        self.h2h1 = nn.Linear(m['rnn_linear1'], m['rnn_linear2']).to(m['device'])
        self.h2h2 = nn.Linear(m['rnn_linear2'], m['rnn_linear3']).to(m['device'])
        self.h2h3 = nn.Linear(m['rnn_linear3'], m['rnn_linear4']).to(m['device'])
        self.h2h4 = nn.Linear(m['rnn_linear4'], m['rnn_linear5']).to(m['device'])

        self.h2o = nn.Linear(m['rnn_linear5'], 2).to(m['device'])
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=m['lr'])

        # visualiser
        self.visualiser = self.Visualiser(self)

    def init_hidden(self, batch_size):
        d = 1
        return torch.zeros(d, batch_size, self.model_params['rnn_nhidden']).to(self.model_params['device'])
        
    def forward(self, data, hidden):
        _, h_t = self.temporal(data, hidden)
        h1 = self.h2h0(h_t)
        h2 = self.h2h1(h1)
        h3 = self.h2h2(h2)
        h4 = self.h2h3(h3)
        h5 = self.h2h4(h4)
        output = self.h2o(h5)

        return output, h_t

    # clips gradient to +-1 to prevent exploding gradients when using reLu
    def clip_gradient(self, max_norm=1):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def forward_step(self, obs, train=True):
        # obs is of shape [batch, seq_len, total_features]
        # typically 16, 70, 2
        try:
            hidden = self.init_hidden(obs.size(0))  # obs.size(0) is the batch size
        except Exception as e:
            print(f"rnn: train_step: hidden initialization error: {e}")
            return 1

        seq_len = obs.size(1)
        losses = []  # List to accumulate losses for each time step
        predictions = []  # (Optional) store predictions if needed
        
        # Create a placeholder for previous output with correct dimensions
        # Will be overwritten with actual values during processing
        prev_output = torch.zeros(obs.size(0), 1, obs.size(2), device=obs.device)  # Use same device as input
        
        # Find first non-NaN data point to initialize prev_output
        found_valid_data = False
        for i in range(seq_len):
            if not torch.isnan(obs[:, i, :]).any():
                prev_output = obs[:, i, :].unsqueeze(1)  # shape: [batch, 1, features]
                found_valid_data = True
                break
                
        if not found_valid_data:
            print("rnn: forward_step: non-valid dataset detected. no instance of non-nan data detected in dataset. Exiting.")
            exit(1)
        
        # Loop over time steps using teacher forcing
        for t in range(seq_len - 1):
            # RNN is given information about the whole sequence
            current_input = obs[:, t, :].unsqueeze(1)  # shape: [batch, 1, features]
            
            # Check if current_input contains NaN values
            if torch.isnan(current_input).any():
                # print(f"manipulating inputs at time {t}")
                # Use the previous output instead, ensuring shape consistency
                current_input = prev_output  # Already has shape [batch, 1, features]
            
            # Forward pass for a single time step
            try:
                # print(f"debug: rnn: forward_step: seq: {t} current_input shape: {current_input.shape}")
                out, hidden = self.forward(current_input, hidden)
                
                # Check if hidden state has NaNs
                if torch.isnan(hidden).any():
                    # print(f"rnn: forward_step: NaN detected in hidden state at step {t}. Resetting hidden state.")
                    hidden = self.init_hidden(obs.size(0))
                
                # print(f"debug: rnn: forward_step: seq: {t} out shape: {out.shape}, sample: {out[0]}")
                
                # Save model prediction for next missing data substitution
                # Reshape the output to match the expected input shape [batch, 1, features]
                prev_output = out.detach().clone().view(obs.size(0), 1, obs.size(2))
                
                if torch.isnan(out).any():
                    print(f"rnn: forward_step: NaN detected in output at step {t}. Exiting.")
                    exit(1)
                # Expected out shape: [batch, 1, output_dim]
            except KeyboardInterrupt:
                print("Training interrupted by user.")
                exit(1)

            # The target for this time step is the next observation in the sequence.
            # NOTE: unslice the last component of the obs tensor to get both charge and time features.
            target = obs[:, t + 1, 0:1].unsqueeze(1)
            
            # Handle target with NaN values - skip loss calculation for this step
            if torch.isnan(target).any():
                # print(f"skipping loss calculation for step {t} due to NaN in target")
                predictions.append(out)
                continue
                
            # Accumulate the loss (using your chosen loss function, e.g., MSELoss)
            # NOTE: also unslice the last component of the out tensor to get both charge and time features.
            # Ensure tensors are on the same device
            loss = self.loss_fn(out[:,:,0:1].to(out.device), target.permute(1, 0, 2).to(out.device))
            losses.append(loss)
            predictions.append(out)

        # Only compute mean loss if we have any valid steps
        if len(losses) > 0:
            avg_loss = torch.stack(losses).mean()
            
            if train:
                # Backpropagation and optimizer step
                self.optimizer.zero_grad()
                avg_loss.backward()
                self.optimizer.step()
                
            return avg_loss.item(), predictions
        else:
            print("No valid steps found for loss calculation")
            return 0.0, predictions

    def train_step(self, traj, time):
        """
        Trains a single batch by making one-step-ahead predictions.
        
        For each time step t in the sequence (except the last one), the RNN is fed the observation
        at time t and outputs a prediction for time t+1. The loss is computed by comparing the prediction
        with the actual observation at time t+1.
        
        Args:
            traj (Tensor): The trajectory data, shape [batch, seq_len, traj_features].
            time (Tensor): The time data, shape [batch, seq_len, time_features].
        
        Returns:
            float: The average loss over the sequence.
        """
        self.train()
        
        # Combine trajectory and time features.
        # Resulting obs shape: [batch, seq_len, total_features]
        obs = torch.cat((traj, time), dim=-1)
        loss, _ = self.forward_step(obs)
        return loss

    def eval_step(self, traj, time, batch_input=True):
        """
        runs the RNN in evaluation mode.
        returns: loss, prediction, target
        """
        # print(f"debug: rnn: eval_step: entered")
        
        # Ensure we're in evaluation mode
        self.temporal.eval()
        
        # Combine trajectory and time features - ensure they're on the same device
        device = traj.device
        # No need for redundant .to(device) if they're already on the same device
        obs = torch.cat((traj, time), dim=-1)
        
        # Get only one batch or use the input as is
        if batch_input:
            obs = obs[0].unsqueeze(0)
        else:
            obs = obs.unsqueeze(0)
            
        # Free up memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Perform forward step without training
        loss, prediction = self.forward_step(obs, train=False)
        return loss, prediction, obs
    
    def train_nepochs(self, n_epochs, model_params=parameters.model_params, dataset=parameters.dataset, records=parameters.records):
        """
        Runs the training loop for n_epochs times where n_epochs is the "epochs per train".

        Arguments:
        ----------
        n_epochs: number of epochs to train for.
        model_params: dictionary of model parameters.
        dataset: dictionary of dataset to train on.

        Returns:
        --------
        n_epochs: number of epochs trained for (equal to n_epochs).
        total_loss_history: equal to MSE loss (below).
        MSE_loss_history: list of MSE loss.
        KL_loss_history: hardcoded to list of 0 (NA).
        """
        # Initialize dataset - use training data with missing points
        data = shjnn.CustomDataset(dataset['train_trajs'], dataset['train_times'])
        # print(f"debug: rnn: train: dataset shape: {len(data)}")
        # Split dataset into training and validation sets
        # TODO: this is technically a good idea but the dataset produces 9 mini-batches in total which makes this split not viable.
        # good to look into this in the future
        train_size = int(0.8 * len(data))  # NOTE: change to 0.8 for 80/20 split
        
        val_size = int(0.2 * len(data))
        train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
        
        # Initialize data loaders
        train_loader = DataLoader(data, batch_size=model_params['n_batch'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

        # Initialize loss history
        val_loss_history = []
        KL_loss_history = []

        print(f"Logs: rnn: train: Training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            try:
                # Clear GPU memory at the start of each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Training phase
                for x_batch, y_batch in train_loader:
                    # No need for redundant device transfers - tensors already moved by DataLoader
                    self.train_step(x_batch, y_batch)
                
                # Validation phase - use original data for evaluation
                epoch_val_loss = 0
                loss_list = []
                
                # Create a validation dataset using the original data (no missing points)
                orig_data = shjnn.CustomDataset(dataset['trajs'], dataset['times'])
                orig_loader = DataLoader(orig_data, batch_size=10, shuffle=False)
                
                with torch.no_grad():  # Disable gradient computation
                    for x_batch, y_batch in orig_loader:
                        # Clear GPU memory between validation batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        _loss, prediction, obs = self.eval_step(x_batch, y_batch)
                        loss_list.append(_loss)
                        
                        # debugging step TODO: remove
                        records['predictions'] = prediction
                        records['targets'] = obs

                # Compute average validation loss for the epoch
                epoch_val_loss = np.mean(loss_list)
                val_loss_history.append(epoch_val_loss)
                self.model_params['loss'].append(epoch_val_loss)
                self.model_params['MSE_loss'].append(epoch_val_loss)
                KL_loss_history.append(0)

                # Print epoch results
                print(f"Epoch {model_params['epochs'] + epoch}: Val Loss = {epoch_val_loss:.4f}")
                
            except KeyboardInterrupt:
                print("Training interrupted by user.")
                return epoch, val_loss_history, val_loss_history, KL_loss_history

        # print average loss:
        print(f"Logs: rnn: train: Average loss: {np.mean(val_loss_history)} at epoch {model_params['epochs'] + epoch}")
        # plot model_params['debug list'] with shape [69, 1, 1, 2]. sequence length of 69 and 2 features.
        # adapt first into [69, 2]
        debug_list = records['predictions']
        debug_list = torch.cat(debug_list, dim=1).squeeze()
        debug_list = debug_list[:, 0].detach().cpu()  # Make sure it's on CPU for plotting
        obs_list = records['targets']
        obs_list = obs_list.squeeze()
        # shape is [1, 70, 2], we need to reduce shape to [69, 2], dropping the first sample.
        obs_list = obs_list[:, 0].detach().cpu()  # Make sure it's on CPU for plotting
        fig, ax = plt.subplots()

        # Plot the data on the axes
        ax.plot(debug_list, label="Prediction", color='r')
        ax.plot(obs_list, label="Observation", color='b')

        # Set title and legend
        ax.set_title("Debug list")
        ax.legend()

        # Display the plot
        # plt.show()
        # Return final epoch and loss histories
        return n_epochs, val_loss_history, val_loss_history, 0
    
    # Implementation for model evaluation using original dataset
    def eval(self, model_params=parameters.model_params, dataset=parameters.dataset):
        """
        Evaluates the model on the original dataset (without missing data)
        
        Returns:
            float: The average loss on the original dataset
        """
        # Use original data without missing points for evaluation
        with torch.no_grad():
            test_dataset = shjnn.CustomDataset(dataset['trajs'], dataset['times'])
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
            
            loss_list = []
            for x_batch, y_batch in test_loader:
                loss, _, _ = self.eval_step(x_batch, y_batch, batch_input=True)
                loss_list.append(loss)
            
            avg_loss = np.mean(loss_list)
            print(f"Logs: rnn: eval: mean loss on original data: {avg_loss} at epoch {model_params['epochs']}.")
            return avg_loss

        # initialise testing data loader for random mini-batches
        # TODO: look at shjnn.dataLoader more.
        test_loader = shjnn.DataLoader(test_dataset, batch_size = model_params['n_batch'], shuffle = True, drop_last = True)
        n = len(test_dataset)

        try:
            for x_batch, y_batch in test_loader:
                _loss = RNN.eval_step(x_batch, y_batch)
                return _loss
        except KeyboardInterrupt:
            return 0
        return 0
    
    def format_output(pred_x, target_timesteps=1000):
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
        # save model params as json file
        folder = model_params['folder']
        epoch = model_params['epochs']
        if not os.path.exists(folder + '/model'):
            os.makedirs(folder + '/model')

        # save model
        # load model params
        optim = model_params['optim']
        loss = model_params['loss']
        epochs = model_params['epochs']
        folder = model_params['folder']
        path = folder + f"/model/save_model_ckpt_{epoch}.pth"
        
        # save rnn model
        torch.save(self.state_dict(), path)
        
        loader.save_model_params(model_params)
        
    def load_model(self, model_params, path):
        self.load_state_dict(torch.load(path))  
    class Visualiser:

        """
        Visualization class for RNN models.
        This class provides methods for visualizing training history, model predictions, 
        and creating animations of the training process for RNN models.
        Attributes
        ----------

        RNN : RNN
            Reference to the parent RNN model instance

        """
        def __init__(self, rnn_instance):
            """
            Initialize the Visualiser with a reference to its parent RNN model.
            
            Parameters
            ----------
            rnn_instance : RNN
                The parent RNN model instance
            """
            self.temporal = rnn_instance

        def plot_training_loss(self, model_params=parameters.model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            visualisation.plot_training_loss(model_params, save=save, split=False, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL, scale='log')

        def display_random_fit(self, model_params=parameters.model_params, dataset=parameters.dataset, show=False, save=True, random_samples=True):
            ''' random assess model fit '''

            # get data
            trajs = dataset['trajs']
            times = dataset['times']
            # print(f"debug: rnn: visualiser: display_random_fit: trajs shape {trajs.shape}, times shape {times.shape}")
            y = dataset['y']

            # get model
            device = model_params['device']
            epoch = model_params['epochs']
            

            # initialise figure
            k = trajs[0].shape[-1]; _w = 7; _h = 4*k; fig = plt.figure(figsize = (_w, _h))
            # fig.canvas.layout.width = '{}in'.format(_w); fig.canvas.layout.height= '{}in'.format(_h)
            ax = [ [ fig.add_subplot(j,1,i) for i in range(1,j+1) ] for j in [k] ][0]
            # concantenate trajs and times
            datas = torch.cat((trajs, times), dim=-1)
            # print(f"debug: rnn: visualiser: display: datas shape: {datas.shape}")
            # select data
            sample_indices = list(range(len(datas)))
            if random_samples:
                random.shuffle(sample_indices)
                # Downsample to avoid cluttering the plot
                sample_indices = sample_indices[::30]
            else:
                sample_indices = model_params['plot']

            # build colourmap
            cnorm  = colors.Normalize(vmin = 0, vmax = len(sample_indices)); smap = cmx.ScalarMappable(norm = cnorm, cmap = 'brg')

            # iterate over transients
            loss_list = []
            for _,i in enumerate(sample_indices):
                
                # get colour
                c = smap.to_rgba(_)
                
                # send mini-batch to device
                traj = trajs[i].view(1, *trajs[i].size()).to(device)
                
                # _time = np.linspace(-7.8, -4.2, 1000)#/10
                # _time = np.linspace(-6.5+6.6, -4.2+6.6, 1000)#/10
                
                # +1 to account for time bias associated with removing the initial rise.
                _time = np.linspace(0, 2.5, 1000) + 1#/10
                
                # _time = np.linspace(-7., -4.2, 1000)
                # _time = np.logspace(-7.8, -4.2, 1000)
                
                # _time = np.logspace(0, 1.7, 20)
                time = torch.Tensor(_time).to(device)

                # perform inference step for prediciton

                loss, prediction, obs = self.temporal.eval_step(datas[i, :, 0].unsqueeze(1), datas[i, :, 1].unsqueeze(1), batch_input=False)
                loss_list.append(loss)
                pred_x = torch.cat(prediction, dim=1)
                pred_x = pred_x[:, :, 0].unsqueeze(2)
                # make RNN output the same format as B-VAE output.
                pred_x = RNN.format_output(pred_x)

                # print(f"debug: rnn: visualiser: display: pred_x shape: {pred_x.shape}")
                pred_x = pred_x.detach().cpu().numpy()[0]

                # return prediction to cpu
                # pred_x = pred_x.cpu().numpy()[0]
                
                # print(pred_x.shape, pred_z[0,0,:])
                
                _traj = trajs[i].detach().cpu()
                _t = times[i].detach().cpu()

                for l in range(k):
                    u = 0
                    
                    # ax[k].set_ylim(-.8, .8)
                    sc_ = 50*1e2/1e3
                    
                    # plot original and predicted trajectories
                    ax[l].plot(_t, _traj[:, l+u]/sc_, '.', alpha = 0.6, color = c)
                    ax[l].plot(_time - 1.0, pred_x[:, l+u]/sc_, '-', label = '{:.1f} J$, {:.1f} V, {:.0e} s'.format(y[i][0], y[i][1], y[i][2]),
                            linewidth = 2, alpha = 0.4, color = c)

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

        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_adaptives(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_adaptive(model_params, dataset, latent_dim_number)
