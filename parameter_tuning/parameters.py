import numpy as np
import torch


import dataloader



# configure initial state here.


# dictionary of hyper params

#contains the dataset. to be loaded in my the dataloader or init.py.
dataset = {
    'trajs': None,
    'times': None,
    'y': None,
    'cut_zero': True # whether to delete all data before t=0
}

#sijin
dataset_PV = {
    'elec_df': r'C:\Users\z5183876\Documents\GitHub\PV-syst-data\Bomen\data_from_server\2021_elec_df.pkl'
    }

#tune hyper-params here.
model_params = {
    # hyper params

    
    'nhidden': 128,
    'rnn_nhidden': 32,
    'obs_dim': 1,

    'latent_dim': 16,
    'lr': 1e-2,
    'n_batch': 16,
    'beta': 4,

    'optim': None,
    'device': None,
    'func': None,
    'rec': None,
    'dec': None,

    # training params
    'total_epochs_train': 4,
    'epochs_per_train': 2,
    'epochs': 0, # a record of the epochs
    'loss': 0, # loss = KL_loss + MSE loss
    'loss_thresh': 500,
    'MSE_loss': 0,
    'KL_loss': 0,

    #labels
    'name': "default",
    'desc': "default run",
    'folder': "saves"
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


# latent_dim, nhidden, rnn_nhidden, obs_dim, lr, n_batch, beta = get_hyper_params(trajs)
# model_params = get_hyper_params(dataset, model_params)
# # func, rec, dec, optim, device, loss, epochs = init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, n_batch, lr)
# model_params = init_model(model_params)
# training_loop(func, rec, dec, optim, trajs, times, n_batch, device, beta)

#clear saves folder
# clear_saves()