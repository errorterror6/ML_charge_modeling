import parameters
from torch import nn
import visualisation

import sys
sys.path.append('../libs/')
import shjnn

class RNN:
    model = None
    criterion = None
    model_params = None

    def __init__(self, model, model_params=parameters.model_params):
        RNN.model = model
        RNN.criterion = nn.MSELoss()
        RNN.model_params = model_params

   
        

    def train_step(traj, time):
        RNN.model.train()
        h = RNN.model.initHidden().to(RNN.model_params['device'])

        # iterate length each trajectory input
        #for t in reversed(range(traj.size(0))):
        for t in reversed(range(traj.size(1))):

            # get trajectory samples in reverse time from final
            #obs = x[t, :].view(1,-1)
            obs = traj[:, t, :]
            #print(obs.shape, obs)

            # run trajectory through recog net
            out, h = RNN.model.forward(obs, h)
        loss = RNN.criterion(out, obs)
        return loss
    
    def train(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):

        dataset = shjnn.CustomDataset(dataset['trajs'], dataset['times'])

        # split dataset into training, validation
        #train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [90, 10])
        train_dataset = dataset

        # initialise training data loader for random mini-batches
        train_loader = shjnn.DataLoader(train_dataset, batch_size = model_params['n_batch'], shuffle = True, drop_last = True)
        n = len(train_dataset)

        loss = []
        print("Logs: rnn: train: printing n_epochs: ", n_epochs)
        for epoch in range(1, n_epochs+1):
            try:
                for x_batch, y_batch in train_loader:
                    _loss = RNN.train_step(x_batch, y_batch)
                    loss.append(_loss.detach().numpy())
            except KeyboardInterrupt:
                return epoch, loss
        epoch_loss = sum(loss) / len(loss)
        return epoch, loss
    
    class visualiser:
        def __init__(self):
            pass

        def plot_training_loss(model_params=parameters.model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL)

        def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
            visualisation.display_random_fit(model_params, dataset, show=show, save=save, random_samples=random_samples)
        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)
            
