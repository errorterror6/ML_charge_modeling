
''' imports '''

# import custom dataset
from .data import CustomDataset, prep_data

# import model components
from .model import init_model

# import training components
from .train import make_train_step

# import inference components
from .infer import make_infer_step

# dimensionality reduction
#from .analysis import get_2d_embedding


import pickle

import numpy as np

from sklearn import preprocessing

import torch
from torch.utils.data import DataLoader



''' orchestration functions '''

def train(func, rec, dec, optim, trajs, times, n_epochs, n_batch, device, beta = None, save_ckpt = None):

    ''' Training Loop

        training loop over epochs

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    ''' initialise dataset and dataloader '''

    # initialise custom dataset
    dataset = CustomDataset(trajs, times)

    # split dataset into training, validation
    #train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [90, 10])
    train_dataset = dataset

    # initialise training data loader for random mini-batches
    train_loader = DataLoader(train_dataset, batch_size = n_batch, shuffle = True, drop_last = True)

    n = len(train_dataset)

    ''' perform training '''

    # create train_step function
    if beta is not None:
        train_step = make_train_step(func, rec, dec, optim, device, beta = beta)
    else:
        train_step = make_train_step(func, rec, dec, optim, device)


    # store for per-batch loss
    loss = []
    MSEloss = []
    KLloss = []

    # iterate each epoch
    for epoch in range(1, n_epochs+1):

        try:

            epoch_loss = 0

            # get batch data from dataloader
            for x_batch, y_batch in train_loader:

                # send mini-batch to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)


                # perform training step, compute loss
                _loss, _px, _kl = train_step(x_batch, y_batch)
                loss.append(_loss)
                #print(_loss, _px, _kl)
                print('Epoch {}: Total Loss {:.3f}, KL Loss {:.2f}, Feat. Loss {:.2f}'.format(epoch, _loss, _kl, -_px))

                # save normalised epoch loss
                epoch_loss += _loss
            epoch_loss /= n

            # display average loss per epoch, total
            print('Epoch {} : {:.2f}, {:.2f}'.format(epoch, epoch_loss, np.median(loss)/n_batch))


            if save_ckpt is not None:
                # save model state
                path = '../models/ckpt_{}_{}.pth'.format(save_ckpt, epoch)
                save_state(path, func, rec, dec, optim, loss, epoch)


        # catch early halt of training
        except KeyboardInterrupt:
            return epoch, loss, MSEloss, KLloss

    # return loss
    return epoch, loss, MSEloss, KLloss



''' save / load state functions '''

def save_state(path, func, rec, dec, optim, loss, epochs):

    ''' Save Model State

        save model and optimiser state

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    torch.save({
        'func_state_dict': func.state_dict(),
        'rec_state_dict': rec.state_dict(),
        'dec_state_dict': dec.state_dict(),
        #'optimizer_state_dict': optim.state_dict(),
        #'loss': loss,
        #'epochs': epochs,
    }, path)



def load_state(path, func, rec, dec, optim, loss, epochs, dev = 'gpu'):

    ''' Load Model State

        load model and optimiser state

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    if dev == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu') )
    else:
        checkpoint = torch.load(path)

    func.load_state_dict(checkpoint['func_state_dict'])
    rec.load_state_dict(checkpoint['rec_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']
    epochs = checkpoint['epochs']


