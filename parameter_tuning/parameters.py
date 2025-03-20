import numpy as np
import torch

import loader

import sys
sys.path.append('../libs/')
import shjnn



# configure initial state here.


# dictionary of hyper params

#contains the dataset. to be loaded in my the dataloader or init.py.
dataset = {
    'trajs': None,
    'times': None,
    'y': None,   # intensity, voltage, delay, thickness.
    'cut_zero': True, # whether to delete all data before t=0
    
    # set indices of missing data here. [1, 70].
    'missing_idx': [4, 14, 26, 34, 50, 60, 65],
    
}

#sijin
dataset_PV = {
    'elec_df': r'C:\Users\z5183876\Documents\GitHub\PV-syst-data\Bomen\data_from_server\2021_elec_df.pkl'
    }

#choose between options 'B-VAE', 'RNN', 'LSTM'
trainer = 'B-VAE'
model = None
rnn = None
b_vae = None

#tune hyper-params here.
model_params = {
    # hyper params

    #NOTE: reccomended to increase rnn_nhidden size.
    'nhidden': 1024,
    'rnn_nhidden': 1024,
    'obs_dim': 1,

    'latent_dim': 16,
    #NOTE: reccomended to decrease this to 1e-5 or similar for training
    'lr': 1e-3,
    'n_batch': 16,  #batch size
    'beta': 0.1,

    'optim': None,
    'device': None,
    'func': None,
    'rec': None,
    'dec': None,

    # training params
    # TODO: doesn't make too much sense for separate epochs per train and total epochs.
    # NOTE: due a bug, total_epochs_Train must be greater than 14.
    'total_epochs_train': 15,
    'epochs_per_train': 15,
    'epochs': [], # a record of the epochs
    'loss': [], # loss = KL_loss + MSE loss
    'loss_thresh': 0.0001,
    'MSE_loss': [],
    'KL_loss': [],
    
    #specifying which trajectories to use (0 - 149 are available)
    'plot': [0, 1, 2, 3, 4],
    'random': False,
    

    #labels
    'name': "default",
    'desc': "default_desc",
    'folder': "saves",

    #free to use memory for debugging only
    'debug': 1,
    'debug_list': [],
    'debug_list2': []
}

records = {
    'loss': [],
    'MSE_loss': [],
    'KL_loss': [],

    #RNN specific
    'predictions': [],
    'targets': []
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

amp = 2
freq = 1
adaptive_training = {
    # change the training parameters during the training procedure, all lists has to have the same length.
    # 'lr': np.logspace(-2, -3, 20),
    'lr': np.logspace(-2.7, -4, 20),
    # 'lr': np.linspace(2e-3, 1e-4, 20),
    'beta': amp*np.sin(np.linspace(0, 5/2*np.pi, 20)) + 2,
    'epochs': [50]*20
}

# options for data visualisation
save = False                     # haven't figured this out yet.
split = False                    #same as above.

# options for grid search

class gridsearch:

    learning_rates = [1e-2]
    n_batches = [16]
    latent_dims = [2]
    betas = [4]

    run_description = "search2"

    data_record = {"learning rate": [],
                    "n_batch": [],
                    "latent_dim": [],
                    "beta": [],
                    "lowest_loss": []}

    loss_record = {"learning rate": [],
                    "n_batch": [],
                    "latent_dim": [],
                    "beta": [],
                    "loss": []}
    
    excel_folder_created = False
    excel_folder_path = ""


# latent_dim, nhidden, rnn_nhidden, obs_dim, lr, n_batch, beta = get_hyper_params(trajs)
# model_params = get_hyper_params(dataset, model_params)
# # func, rec, dec, optim, device, loss, epochs = init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, n_batch, lr)
# model_params = init_model(model_params)
# training_loop(func, rec, dec, optim, trajs, times, n_batch, device, beta)

#clear saves folder
# clear_saves()