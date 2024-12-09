import torch
import matplotlib.pyplot as plt

import parameters
import sys
sys.path.append('../libs/')
import shjnn


def training_loop(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
    ''' run training loop with save 
        args: n_epochs, model_params, dataset
        return: model_params (updated)
    '''

    print("Logs: Training: Running training loop")
    

    # beta for beta latent dissentanglement
    #beta = 4.
    # beta = .01

    # update learning rate
    # lr = 1e-3

    lr = model_params['lr']
    n_batch = model_params['n_batch']
    beta = model_params['beta']
    optim = model_params['optim']


    for g in optim.param_groups:
        g['lr'] = lr

    # get data
    trajs = dataset['trajs']
    times = dataset['times']

    # get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    device = model_params['device']

    # run training for epochs, return loss
    try:
        _epochs, _loss, _MSE_loss, _KL_loss = shjnn.train(func, rec, dec, optim, trajs[:], times[:], n_epochs, n_batch, device, beta)
        print('Try')
        print('loss', _loss, 'epochs', _epochs, 'MSE_loss', _MSE_loss, 'KL_loss', _KL_loss)
        # plot all three losses
        plt.figure()
        plt.plot(_loss, label = 'loss')
        plt.plot(_MSE_loss, label = 'MSE_loss')
        plt.plot(_KL_loss, label = 'KL_loss')
        plt.legend()
        plt.show()

        # update loss, epochs
        print(type(model_params['loss']), type(_loss))
        print(type(model_params['MSE_loss']), type(_MSE_loss))
        print(type(model_params['KL_loss']), type(_KL_loss))
        model_params['loss'] += _loss
        model_params['epochs'] += _epochs
        model_params['MSE_loss'] += _MSE_loss
        model_params['KL_loss'] += _KL_loss

        # plot all three losses again
        plt.figure()
        plt.plot(model_params['loss'], label = 'loss')
        # plt.plot(model_params['MSE_loss'], label = 'MSE_loss')
        # plt.plot(model_params['KL_loss'], label = 'KL_loss')
        plt.legend()
        plt.show()
    except:
        model_params['loss'] += model_params['loss'][-1]
        model_params['epochs'] += model_params['epochs'][-1]
        model_params['MSE_loss'] += model_params['MSE_loss'][-1]
        model_params['KL_loss'] += model_params['KL_loss'][-1]
        print('Except')
        print('loss', model_params['loss'], 'epochs', model_params['epochs'], 'MSE_loss', model_params['MSE_loss'], 'KL_loss', model_params['KL_loss'])

    print("Logs: Training: Finished training loop")
    return model_params

def done_training(model_params=parameters.model_params):
    """ check if training is done
        args: model_params
        return: bool
    """
    reach_epoch = model_params['epochs'] >= model_params['total_epochs_train']
    if len(model_params['loss']) < 10:
        reach_loss = False
    elif model_params['loss'] == None:
        reach_loss = False
    else:
        reach_loss = model_params['loss'][-1] < model_params['loss_thresh']
    return  reach_epoch or reach_loss

#chage beta to a dynamic rate
def update_beta(model_params, epochs):
    model_params['beta'] = model_params['beta']

#chage lr to a dynamic rate
def update_lr(model_params, epochs):
    model_params['lr'] = model_params['lr']

def update_n_epochs(model_params, epochs):
    return

def update_params(model_params, epochs):
    update_beta(model_params, epochs)
    update_lr(model_params, epochs)
    update_n_epochs(model_params, epochs)