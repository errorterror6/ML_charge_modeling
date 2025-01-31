import parameters
from torch import nn
import torch, random, glob, os
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
            )
        #TODO: encode experimental variables into the hidden layer as init.
        #hidden to output (y_t+1, time_t+1)
        self.h2o = nn.Linear(m['nhidden'], 2)
        
      
    def forward(self, data):
        _, hidden = self.rnn(data)
        output = self.h2o(hidden)
        return output

    #clips gradient to +-1 to prevent exploding gradients when using reLu
    def clip_gradient(self, max_norm=1):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        

    def train_step(self, traj, time):
        self.rnn.train()
        total_loss = 0  # Accumulate loss over all time steps

        # Initialize hidden state (if using a vanilla RNN, LSTM, or GRU)
        hidden = self.rnn.init_hidden(traj.size(0))  # Batch size is traj.size(0)

        # Iterate over each time step in the trajectory
        for t in range(traj.size(1) - 1):  # Stop at the second-to-last time step
            # Concatenate trajectory and time at each time step
            obs = torch.cat((traj[:, t, :], time[:, t, :]), dim=-1)

            # Run trajectory through the RNN
            out, hidden = self.forward(obs, hidden)  # Pass hidden state

            # Prepare the target (next time step's values)
            target = torch.cat((traj[:, t+1, :], time[:, t+1, :]), dim=-1)

            # Compute the loss
            loss = self.criterion(out, target)
            total_loss += loss  # Accumulate loss

        # Return the average loss over all time steps
        return total_loss / (traj.size(1) - 1)

    def eval_step(traj, time):
        with torch.no_grad():
            RNN.model.eval()
            h = RNN.model.initHidden().to(RNN.model_params['device'])

            # iterate length each trajectory input
            #for t in reversed(range(traj.size(0))):
            # TODO： put the time step into the RNN, unreverse this.
            for t in range(traj.size(1)):

                # get trajectory samples in reverse time from final
                #obs = x[t, :].view(1,-1)
                obs = traj[:, t, :]
                # print("obs shape:;" , obs.shape, obs)

                # run trajectory through recog net
                out, h = RNN.model.forward(obs, h)
            loss = RNN.criterion(out, obs)
        return loss
    
    '''
    runs the training loop for n_epochs times where n_epochs is the "epochs per train".
    '''
    def train(self, n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
        
        dataset = shjnn.CustomDataset(dataset['trajs'], dataset['times'])
        # TODO: TRAJS should have all of the experimental variables (5 of em)

        # split dataset into training, validation
        #train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [90, 10])
        train_dataset = dataset

        # initialise training data loader for random mini-batches
        #TODO: look at shjnn.dataLoader more.
        train_loader = shjnn.DataLoader(train_dataset, batch_size = model_params['n_batch'], shuffle = True, drop_last = True)
        n = len(train_dataset)

        loss = []
        print("Logs: rnn: train: printing n_epochs: ", n_epochs)
        for epoch in range(1, n_epochs+1):
            try:
                for x_batch, y_batch in train_loader:
                    _loss = self.train_step(x_batch, y_batch)
                    # loss.append(_loss.detach().numpy())
                    loss = 0   #TODO: placeholder.
            except KeyboardInterrupt:
                return epoch, loss
        epoch_loss = sum(loss) / len(loss)
        return epoch, loss
    
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

