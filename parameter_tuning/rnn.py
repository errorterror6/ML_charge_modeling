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


import sys
sys.path.append('../libs/')
import shjnn

class RNN(nn.Module):

    def __init__(self, m=parameters.model_params):
        super().__init__()
        
        self.model_params = m
        self.criterion = nn.MSELoss()
        #input : (y_t, time_t) to hidden
        #TODO: ensure that the starting params of the RNN are the same run to run.
        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=m['rnn_nhidden'],
            #use relu + clip_gradient if poor results with tanh
            nonlinearity='tanh',
            device=m['device'],
            batch_first=True
            )
        #TODO: encode experimental variables into the hidden layer as init.
        #hidden to output (y_t+1, time_t+1)
        self.h2h1 = nn.Linear(m['nhidden'], m['nhidden'])

        self.h2o = nn.Linear(m['nhidden'], 2)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=m['lr'])

        #visualiser
        self.visualiser = self.Visualiser(self)



    def init_hidden(self, batch_size):
        d = 1
        return torch.zeros(d, batch_size, self.model_params['nhidden'])
        
      
    def forward(self, data, hidden):
        _, h_t = self.rnn(data, hidden)
        h2 = self.h2h1(h_t)
        output = self.h2o(h2)

        # output = self.h2o(h_t)
        return output, h_t

    #clips gradient to +-1 to prevent exploding gradients when using reLu
    def clip_gradient(self, max_norm=1):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def forward_step(self, obs, train=True):
        #obs is of shape [batch, seq_len, total_features]
        #ypically 16, 70, 2
        try:
            hidden = self.init_hidden(obs.size(0))  # obs.size(0) is the batch size
        except Exception as e:
            print(f"rnn: train_step: hidden initialization error: {e}")
            return 1

        seq_len = obs.size(1)
        losses = []  # List to accumulate losses for each time step
        predictions = []  # (Optional) store predictions if needed

        # Loop over time steps using teacher forcing.
        # For t = 0 to seq_len - 2:
        #   - Use obs[:, t, :] as input.
        #   - Predict obs[:, t+1, :].
        it = 0
        for t in range(seq_len - 1):
            # RNN is given information about the whole sequence.
            current_input = obs[:, t, :].unsqueeze(1)

            # Forward pass for a single time step
            try:
                out, hidden = self.forward(current_input, hidden)
                # Expected out shape: [batch, 1, output_dim]
            except KeyboardInterrupt:
                print("Training interrupted by user.")
                exit(1)

            # The target for this time step is the next observation in the sequence.
            target = obs[:, t + 1, :].unsqueeze(1)
            # Accumulate the loss (using your chosen loss function, e.g., MSELoss)
            loss = self.loss_fn(out, target.permute(1, 0, 2))
            losses.append(loss)
            predictions.append(out)

        avg_loss = torch.stack(losses).mean()
        if train:
        # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()

        return avg_loss.item(), predictions

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
        self.rnn.train()
        
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
        #TODO: change so that eval step only runs one of the batches...
       
        self.rnn.eval()
        # Combine trajectory and time features.
        # Resulting obs shape: [batch, seq_len, total_features]
        obs = torch.cat((traj, time), dim=-1)
        #get only one batch
        if batch_input:
            obs = obs[0].unsqueeze(0)
        else:
            obs = obs.unsqueeze(0)
        loss, prediction = self.forward_step(obs, train=False)
        return loss, prediction, obs

        # Return the average loss over all time steps
    
    def train(self, n_epochs, model_params=parameters.model_params, dataset=parameters.dataset, records=parameters.records):
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
        # Initialize dataset
        data = shjnn.CustomDataset(dataset['trajs'], dataset['times'])

        # Split dataset into training and validation sets
        # TODO: this is technically a good idea but the dataset produces 9 mini-batches in total which makes this split not viable.
        # good to look into this in the future
        # train_size = int(0.9 * len(data))  # 90% training, 10% validation
        # val_size = len(data) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
        
        #TODO: temporary code addressing the above
        train_dataset = data

        # Initialize data loaders
        train_loader = DataLoader(train_dataset, batch_size=model_params['n_batch'], shuffle=True, drop_last=True)
        # TODO: begin temporary code. Extract the first batch
        first_batch = next(iter(train_loader))
        inputs, targets = first_batch

        # Create val_loader
        val_dataset = TensorDataset(inputs, targets)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        #TODO: part of the above TODO's. above line of code is temporary fix.
        # val_loader = shjnn.DataLoader(val_dataset, batch_size=model_params['n_batch'], shuffle=False, drop_last=True)

        # Initialize loss history
        val_loss_history = []
        KL_loss_history = []

        print(f"Logs: rnn: train: Training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            try:
                # Training phase

                for x_batch, y_batch in train_loader:
                    self.train_step(x_batch, y_batch)
                # Compute average training loss for the epoch

                # Validation phase
                epoch_val_loss = 0
                loss_list = []
                with torch.no_grad():  # Disable gradient computation
                    for x_batch, y_batch in val_loader:
                        _loss, prediction, obs = self.eval_step(x_batch, y_batch)
                        loss_list.append(_loss)
                        #debugging step TODO: remove
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
        #adapt first into [69, 2]
        debug_list = records['predictions']
        debug_list = torch.cat(debug_list, dim=1).squeeze()
        debug_list = debug_list[:, 0]
        obs_list = records['targets']
        obs_list = obs_list.squeeze()
        #shape is [1, 70, 2], we need to reduce shape to [69, 2], dropping the first sample.
        obs_list = obs_list[:, 0]
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
    
    #TODO: not implemented. partial implementation embedded in train, and eval_step.
    def eval(model_params=parameters.model_params, dataset=parameters.dataset):
        return
        dataset = shjnn.CustomDataset(dataset['trajs'], dataset['times'])
        test_dataset = dataset

        # initialise testing data loader for random mini-batches
        #TODO: look at shjnn.dataLoader more.
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
            self.RNN = rnn_instance

        def plot_training_loss(self, model_params=parameters.model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            visualisation.plot_training_loss(model_params, save=save, split=False, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL, scale='log')

        def display_random_fit(self, model_params=parameters.model_params, dataset=parameters.dataset, show=False, save=True, random_samples=True):
            ''' random assess model fit '''


            # get data
            trajs = dataset['trajs']
            times = dataset['times']
            print(f"debug: rnn: visualiser: display_random_fit: trajs shape {trajs.shape}, times shape {times.shape}")
            y = dataset['y']

            # get model
            device = model_params['device']
            epoch = model_params['epochs']
            

            # initialise figure
            k = trajs[0].shape[-1]; _w = 7; _h = 4*k; fig = plt.figure(figsize = (_w, _h))
            #fig.canvas.layout.width = '{}in'.format(_w); fig.canvas.layout.height= '{}in'.format(_h)
            ax = [ [ fig.add_subplot(j,1,i) for i in range(1,j+1) ] for j in [k] ][0]
            #concantenate trajs and times
            datas = torch.cat((trajs, times), dim=-1)
            print(f"debug: rnn: visualiser: display: datas shape: {datas.shape}")
            # select data
            j = list(range(len(datas)))
            if random_samples:
                random.shuffle(j)

            # downsample
            j = j[::30]

            # build colourmap
            cnorm  = colors.Normalize(vmin = 0, vmax = len(j)); smap = cmx.ScalarMappable(norm = cnorm, cmap = 'brg')

            # iterate over transients
            loss_list = []
            for _,i in enumerate(j):
                
                # get colour
                c = smap.to_rgba(_)
                
                # send mini-batch to device
                traj = trajs[i].view(1, *trajs[i].size()).to(device)
                
                #_time = np.linspace(-7.8, -4.2, 1000)#/10
                #_time = np.linspace(-6.5+6.6, -4.2+6.6, 1000)#/10
                
                #+1 to account for time bias associated with removing the initial rise.
                _time = np.linspace(0, 2.5, 1000) + 1#/10
                
                #_time = np.linspace(-7., -4.2, 1000)
                #_time = np.logspace(-7.8, -4.2, 1000)
                
                #_time = np.logspace(0, 1.7, 20)
                time = torch.Tensor(_time).to(device)

                # perform inference step for prediciton


                loss, prediction, obs = self.RNN.eval_step(datas[i, :, 0].unsqueeze(1), datas[i, :, 1].unsqueeze(1), batch_input=False)
                loss_list.append(loss)
                pred_x = torch.cat(prediction, dim=1)
                pred_x = pred_x[:, :, 0].unsqueeze(2)
                #make RNN output the same format as B-VAE output.
                pred_x = RNN.format_output(pred_x)

                # print(f"debug: rnn: visualiser: display: pred_x shape: {pred_x.shape}")
                pred_x = pred_x.cpu().detach().numpy()[0]

                # return prediction to cpu
                # pred_x = pred_x.cpu().numpy()[0]
                
                #print(pred_x.shape, pred_z[0,0,:])
                
                _traj = trajs[i].cpu()
                _t = times[i].cpu()

                for l in range(k):
                    u = 0
                    
                    #ax[k].set_ylim(-.8, .8)
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

            #plt.xscale('log')
            plt.legend(loc='upper right', title='Intensity, Bias, Delay')

            plt.tight_layout()

            if show:
                plt.show()
            if save:
                #save as a png]
                #todo: change
                folder = model_params['folder']
                #if saves folder does not exist create it
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig(f"{folder}/training_epoch_{epoch:04d}.png", dpi=300)

        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_adaptives(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_adaptive(model_params, dataset, latent_dim_number)
            