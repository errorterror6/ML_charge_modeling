import parameters
import loader

import sys
sys.path.append('../libs/')
import shjnn



def init_shjnn(model_params=parameters.model_params):
    """
        inits models - always to CPU.
        args: model_params
        return: model_params (updated)
    """
    print("Logs: Init: Initialising SHJNN library")
    latent_dim = model_params['latent_dim']
    nhidden = model_params['nhidden']
    rnn_nhidden = model_params['rnn_nhidden']
    obs_dim = model_params['obs_dim']
    lr = model_params['lr']
    n_batch = model_params['n_batch']

    func, rec, dec, optim, device = shjnn.init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, n_batch, lr, device = 'cpu')
    loss: list = []
    epochs = 0
    device

    model_params['func'] = func
    model_params['rec'] = rec
    model_params['dec'] = dec
    model_params['optim'] = optim
    model_params['device'] = device
    model_params['loss'] = loss
    model_params['epochs'] = epochs
    model_params['loss'] = loss
    #TODO: review
    model_params['MSE_loss'] = loss
    model_params['KL_loss'] = loss

    print("Logs: Init: Finished initialising SHJNN library")

    return model_params

def load_data():
    print("Logs: Init: Initialising data from dataloader")
    loader.load_data()
    trajs = parameters.dataset['trajs']
    obs_dim = trajs[0].size()[1]
    parameters.model_params['obs_dim'] = obs_dim
    print("Logs: Init: Finished initialising data from dataloader")

