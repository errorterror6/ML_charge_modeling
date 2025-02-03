import parameters
from torch import nn
import torch, random, glob, os
from torch.utils.data import DataLoader, TensorDataset
import visualisation
import matplotlib.pyplot as plt
import numpy as np
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
            hidden_size=m['nhidden'],
            #use relu + clip_gradient if poor results with tanh
            nonlinearity='tanh',
            device=m['device'],
            batch_first=True
            )
        #TODO: encode experimental variables into the hidden layer as init.
        #hidden to output (y_t+1, time_t+1)
        self.h2o = nn.Linear(m['nhidden'], 2)

    def init_hidden(self, batch_size):
        return torch.zeros(sequence_length, batch_size, self.model_params['nhidden'])
        
      
    def forward(self, data, hidden):
        _, h_t = self.rnn(data, hidden)
        output = self.h2o(h_t)
        return output, h_t

    #clips gradient to +-1 to prevent exploding gradients when using reLu
    def clip_gradient(self, max_norm=1):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        

    def train_step(self, traj, time):
        """
        trains a single batch, composed of samples
        """
        self.rnn.train()
        total_loss = 0  # Accumulate loss over all time steps

        # Initialize hidden state (if using a vanilla RNN, LSTM, or GRU)
        
        try:
            hidden = self.init_hidden(traj.size(0))  # Batch size is traj.size(0)
        except Exception as e:
            print(f"rnn: train_step: hidden except: Error: {e}")
            return 1
        # one sequence at a time.

        # Iterate over each time step in the trajectory
        # TODO: remove for loop
        for t in range(traj.size(1) - 1):  # Stop at the second-to-last time step
            # Concatenate trajectory and time at each time step
            # TODO: do the concantenation above the loop.
            obs = torch.cat((traj[:, t, :], time[:, t, :]), dim=-1).unsqueeze(1)
            # shape: 16, sequence length, data
            print(f"obs shape: {obs.shape}")
            # Run trajectory through the RNN
            try:
                out, hidden = self.forward(obs, hidden)  # Pass hidden state
                # print(f"logs: rnn: train_step: obs: {obs}")
                # print(f"logs: rnn: train_step: out: {out}")
            except Exception as e:
                print(f"rnn: train_step: except: Error: {e}")
                return 1

            # Prepare the target (next time step's values)
            # TODO: pulled above the loop
            target = torch.cat((traj[:, t+1, :], time[:, t+1, :]), dim=-1).unsqueeze(1)

            # Compute the loss
            loss = self.criterion(out, target)
            total_loss += loss  # Accumulate loss

        # Return the average loss over all time steps
        return total_loss / (traj.size(1) - 1)

    def eval_step(self, traj, time):
        try:
            self.rnn.eval()
            total_loss = 0  # Accumulate loss over all time steps

            # Initialize hidden state (if using a vanilla RNN, LSTM, or GRU)
            
            try:
                hidden = self.init_hidden(traj.size(0))  # Batch size is traj.size(0)
            except Exception as e:
                print(f"rnn: train_step: hidden except: Error: {e}")
                return 1
            # one sequence at a time.

            # Iterate over each time step in the trajectory
            for t in range(traj.size(1) - 1):  # Stop at the second-to-last time step
                # Concatenate trajectory and time at each time step
                obs = torch.cat((traj[:, t, :], time[:, t, :]), dim=-1).unsqueeze(1)

                # Run trajectory through the RNN
                try:
                    out, hidden = self.forward(obs, hidden)  # Pass hidden state
                except Exception as e:
                    print(f"rnn: train_step: except: Error: {e}")
                    return 1

                # Prepare the target (next time step's values)
                target = torch.cat((traj[:, t+1, :], time[:, t+1, :]), dim=-1).unsqueeze(1)

                # Compute the loss
                loss = self.criterion(out, target)
                total_loss += loss  # Accumulate loss

            # Return the average loss over all time steps
            return total_loss / (traj.size(1) - 1)
        except Exception as e:
            print(f"rnn: eval_step: except: Error: {e}")
            return 1
    
    
    def train(self, n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
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
                self.rnn.train()  # Set model to training mode
                epoch_train_loss = 0
                for x_batch, y_batch in train_loader:
                    _loss = self.train_step(x_batch, y_batch)
                    epoch_train_loss += _loss.item()  # Accumulate batch loss

                # Compute average training loss for the epoch
                #TODO: question: is this needed?
                epoch_train_loss /= len(train_loader)

                # Validation phase
                self.rnn.eval()  # Set model to evaluation mode
                epoch_val_loss = 0
                with torch.no_grad():  # Disable gradient computation
                    for x_batch, y_batch in val_loader:
                        _loss = self.eval_step(x_batch, y_batch)
                        epoch_val_loss += _loss.item()  # Accumulate batch loss

                # Compute average validation loss for the epoch
                epoch_val_loss /= len(val_loader)
                val_loss_history.append(epoch_val_loss)
                KL_loss_history.append(0)

                # Print epoch results
                print(f"Epoch {model_params["epochs"] + epoch}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")
                
            except KeyboardInterrupt:
                print("Training interrupted by user.")
                return epoch, val_loss_history, val_loss_history, KL_loss_history
            
            except Exception as e:
                print(f"rnn: train: except: Error: {e}")
                return epoch, val_loss_history, val_loss_history, KL_loss_history

        # print average loss:
        print(f"Logs: rnn: train: Average loss: {np.mean(val_loss_history)} at epoch {model_params['epochs'] + epoch}")
        
        # Return final epoch and loss histories
        return n_epochs, val_loss_history, val_loss_history, 0
    
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
        
    
    class visualiser:
        def __init__(self):
            pass

        def plot_training_loss(model_params=parameters.model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL)

        def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
            pass

        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)

