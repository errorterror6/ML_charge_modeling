import torch
import matplotlib.pyplot as plt
import numpy as np
import time, os
import pandas as pd


import init
import loader
import visualisation
import parameters
import sys
from serial import RNN
from b_vae import B_VAE
sys.path.append('../../libs/')
import shjnn


def B_VAE_training_loop(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
    ''' run training loop with save 
        args: n_epochs, model_params, dataset
        return: model_params (updated)
    '''    

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

    # get training data with missing points
    trajs = dataset['train_trajs']
    times = dataset['train_times']

    # get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    device = model_params['device']

    # run training for epochs, return loss
    # try:
    _epochs, _loss, _MSE_loss, _KL_loss = shjnn.train(func, rec, dec, optim, trajs[:], times[:], n_epochs, n_batch, device, beta)
    print('Logs: training: B_VAE_training_loop: Try')
    # print('loss', _loss, 'epochs', _epochs, 'MSE_loss', _MSE_loss, 'KL_loss', _KL_loss)
    
    
    # Evaluate on original dataset (without missing points)
    eval_loss = parameters.model.eval(model_params, dataset)

    # update loss, epochs
    model_params['epochs'] += _epochs
    # print(f"debug: printing out _epochs: {_epochs}")
    model_params['loss'].append(np.average(eval_loss))
    # print(f"debug: printing out mse loss: {np.average(_MSE_loss)}")
    model_params['MSE_loss'].append(np.average(_MSE_loss))
    model_params['KL_loss'].append(np.average(_KL_loss))

    # except:
        
    #     model_params['epochs'] += model_params['epochs'][-1]
    #     model_params['loss'].append(model_params['loss'][-1])
    #     model_params['MSE_loss'].append(model_params['MSE_loss'][-1])
    #     model_params['KL_loss'].append(model_params['KL_loss'][-1])
    #     print('==================================')
    #     print('Logs: training: B_VAE_training_loop: Exception raised, shjnn failed.')
    #     print('loss', model_params['loss'], 'epochs', model_params['epochs'], 'MSE_loss', model_params['MSE_loss'], 'KL_loss', model_params['KL_loss'])
    #     print('==================================')

    return model_params

def RNN_training_loop(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
    ''' run training loop with save 
        args: n_epochs, model_params, dataset
        return: model_params (updated)
    '''    

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

    # get training data with missing points
    trajs = dataset['train_trajs']
    times = dataset['train_times']

    # get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    device = model_params['device']

    # run training for epochs, return loss
    rnn = parameters.model
    _epochs, _loss, _, _ = rnn.train_nepochs(n_epochs, model_params=model_params, dataset=dataset)

    # print('loss', _loss, 'epochs', _epochs, 'MSE_loss', _MSE_loss, 'KL_loss', _KL_loss)

    # update loss, epochs
    model_params['epochs'] += _epochs
    # model_params['loss'].append(np.average(_loss))

    return model_params

def LSTM_training_loop(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):
    ''' run training loop with save 
        args: n_epochs, model_params, dataset
        return: model_params (updated)
    '''    

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

    # get training data with missing points
    trajs = dataset['train_trajs']
    times = dataset['train_times']

    # get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    device = model_params['device']

    # run training for epochs, return loss
    lstm = parameters.model
    _epochs, _loss, _, _ = lstm.train_nepochs(n_epochs, model_params=model_params, dataset=dataset)

    print('Logs: training: lstm_training_loop: Try')
    # print('loss', _loss, 'epochs', _epochs, 'MSE_loss', _MSE_loss, 'KL_loss', _KL_loss)

    # update loss, epochs
    model_params['epochs'] += _epochs
    # model_params['loss'].append(np.average(_loss))
    print(f"debug: loss size: {len(model_params['loss'])}")

    return model_params

def done_training(model_params=parameters.model_params):
    """ check if training is done
        args: model_params
        return: bool
    """
    print("logs: Training: done_training with epochs: " + str(model_params['epochs']) + " out of total epochs: " + str(model_params['total_epochs_train']))
    reach_epoch = model_params['epochs'] >= model_params['total_epochs_train']
    if len(model_params['loss']) < 10:
        reach_loss = False
    elif model_params['loss'] == None:
        reach_loss = False
    else:
        reach_loss = model_params['loss'][-1] < model_params['loss_thresh']
    print("logs: Training: done_training: " + str(reach_epoch or reach_loss))
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

def train(model_params, dataset, grid_search=False, grid_search_name="default"):
    print("logs: Training: Running training")
    print('The model parameters are: ' + str(model_params))

    name = model_params['name']
    desc = model_params['desc']

    #replace all spaces with _
    name = name.replace(" ", "_")
    desc = desc.replace(" ", "_")

    # create folder based on time, name and description
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder = './saves/' + timestr + '_' + name + '_' + desc
    if grid_search:
        folder = f"./saves/grid_seach/{grid_search_name}/" + timestr + '_' + name + '_' + desc
        #make parent dirs
        # if not os.path.exists(f"./saves/grid_seach"):
        #     os.makedirs(f"./saves/grid_seach")
        # if not os.path.exists(f"./saves/grid_seach/{grid_search_name}"):
        #     os.makedirs(f"./saves/grid_seach/{grid_search_name}")
    model_params['folder'] = folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    train_epochs = model_params['epochs_per_train']

    while not done_training(model_params):
        match parameters.trainer:
            case 'B-VAE':
                B_VAE_training_loop(train_epochs, model_params, dataset)
                
            case 'RNN':
                RNN_training_loop(train_epochs, model_params, dataset)
            case 'LSTM':   
                LSTM_training_loop(train_epochs, model_params, dataset)
                
                # print("Logs: training: train: debug: loss: ", parameters.model_params['loss'])
            case _:
                print("Error: Invalid trainer, pick from options 'B-VAE'. Exiting.")
                exit(1)
        update_params(model_params, model_params['epochs'])
        # loader.save_random_fit(model_params, dataset, random_samples=False)
        # loader.save_model(model_params)
        parameters.model.visualiser.display_random_fit(show=False, random_samples=model_params['random'])
    parameters.model.visualiser.plot_training_loss(model_params, save=True, split=False)
    # loader.save_model_params(model_params)
    parameters.model.save_model(model_params)
    # parameters.model.Visualiser.compile_learning_gif(model_params, display=False)
    print("logs: Training: Finished training")
    return model_params

def adaptive_run_and_save(model_params, dataset, adaptive_training):
    ''' run and save model with adaptive training '''
    # print out hte adaptive_training parameters
    print("logs: Training: Running adaptive training")
    print('The adaptive training parameters are: ' + str(adaptive_training))

    # extract the epochs
    epochs_list = adaptive_training['epochs']
    # extract the learning rates
    lr_list = adaptive_training['lr']
    # extract the beta
    beta_list = adaptive_training['beta']

    # update the total epochs to the sum of the list of epochs
    model_params['total_epochs_train'] = sum(epochs_list)

    # initialization
    name = model_params['name']
    desc = 'adaptive run'

    #replace all spaces with _
    name = name.replace(" ", "_")
    desc = desc.replace(" ", "_")

    # create folder based on time, name and description
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder = './saves/' + timestr + '_' + name + '_' + desc
    model_params['folder'] = folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    init.init_shjnn(model_params)

    # define cumulative epoch counts for plotting
    cumulative_epochs = np.cumsum(epochs_list)
    # plot the lr and beta as a function of epoch
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    ax[0].plot(cumulative_epochs, lr_list, '-o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Learning Rate')
    ax[1].plot(cumulative_epochs, beta_list, '-o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Beta')
    # save the figure
    folder = model_params['folder']
    save_folder = f"{folder}/adaptive_parameters/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(f"{save_folder}/adaptive_parameters.png")
    #plt.show()

    # for each epoch, learning rate and beta, run the model and save
    for k in range(len(epochs_list)):

        # update the model parameters
        model_params['lr'] = lr_list[k]
        model_params['beta'] = beta_list[k]
        train_epochs = epochs_list[k]
        update_params(model_params, model_params['epochs'])
        
        # conduct training
        #TODO: review if possible to redirect this to "train" function.
        match parameters.trainer:
            case 'B-VAE':
                B_VAE_training_loop(train_epochs, model_params, dataset)
            case _:
                print("Error: Invalid trainer, pick from options 'B-VAE'. Exiting.")
                exit(1)
        print("epochs printout:   ")
        print(model_params['epochs'])
        print("loss printout: ")
        print(model_params['loss'])
        update_params(model_params, model_params['epochs'])
        loader.save_random_fit(model_params, dataset, random_samples=False)
        loader.save_model(model_params)
        loader.save_model_params(model_params)
        visualisation.plot_training_loss(model_params, save=True ,split=True)
        visualisation.compile_learning_gif(model_params, display=False)

        # check if the training is done
        if done_training(model_params):
            print('Training is done')
            break
    print("logs: Training: Finished adaptive training")

def grid_search(model_params):
    print("logs: Training: Running grid search")
    learning_rates = parameters.gridsearch.learning_rates
    n_batches = parameters.gridsearch.n_batches
    latent_dims = parameters.gridsearch.latent_dims
    betas = parameters.gridsearch.betas


    # beta never above 4  done
    # latent_dims : 1, 2, 4, 8, 16   done
    # learning rats: 1e-2, 5e-3, 1e-3, 5e-4, 1e-4    done
    # ode solver struggles to "converge" when its already close. implement time/loss wrapper that stops the running when reached and output what epoch it reached the limit.

    #check both MSE and KL. KL should be around 1, MSE is dependent on the data. Try to normalise MSE so that it is around KL. feat. loss is MSE
    # check if kl loss is before or after beta.
    # could plot the KL loss and the MSE loss on the same graph to compare and see how its training.

    # add training gif for each run. done
    # save model checkpoints.   done
    

    # learning_rates = [5e-3, 1e-3]
    # n_batches = [16]
    # latent_dims = [8]
    # betas = [.001, 4,]

    run_description = parameters.gridsearch.run_description

    data_record = parameters.gridsearch.data_record
    
    loss_record = parameters.gridsearch.loss_record

    total_runs = len(learning_rates) * len(n_batches) * len(latent_dims) * len(betas)
    excel_folder_created = parameters.gridsearch.excel_folder_created
    excel_folder_path = parameters.gridsearch.excel_folder_path

    
    for lr in learning_rates:
        for n_batch in n_batches:
            for latent_dim in latent_dims:
                for beta in betas:
                    model_params['lr'] = lr
                    model_params['n_batch'] = n_batch
                    model_params['latent_dim'] = latent_dim
                    model_params['beta'] = beta
                    model_params['desc'] = f"lr_{lr}_nb_{n_batch}_ld_{latent_dim}_b_{beta}"

                    try:
                        run_and_save(model_params, parameters.dataset, grid_search=True, grid_search_name=run_description)
                    except AssertionError:
                        model_params["loss"].append(-1)
                    # handle assertion error
                    

                    #write summary data to excel sheet
              

                    

                    # append information to data_record
                    data_record["learning rate"].append(lr)
                    data_record["n_batch"].append(n_batch)
                    data_record["latent_dim"].append(latent_dim)
                    data_record["beta"].append(beta)
                    loss = model_params['loss']
                    lowest_loss = min(loss)
                    data_record["lowest_loss"].append(lowest_loss)

                    # append information to loss_record
                    loss_record["learning rate"].append(lr)
                    loss_record["n_batch"].append(n_batch)
                    loss_record["latent_dim"].append(latent_dim)
                    loss_record["beta"].append(beta)
                    loss_record["loss"].append(loss)

                    model_params["epochs"] = 0
                    model_params["loss"] = 0

    if not excel_folder_created:
    #create excel sheet
        folder = model_params['folder']
        excel_folder_path = f"{folder}/../excel_output"
        if not os.path.exists(excel_folder_path):
            os.makedirs(excel_folder_path)
        excel_folder_created = True

    #write to excel
    df = pd.DataFrame(data_record)
    sheet = pd.ExcelWriter(f"{excel_folder_path}/summary.xlsx")
    df.to_excel(sheet)
    sheet.close()

    lf = pd.DataFrame(loss_record)
    loss_sheet = pd.ExcelWriter(f"{excel_folder_path}/loss_record.xlsx")
    lf.to_excel(loss_sheet)
    loss_sheet.close()
    print("logs: Training: Finished grid search")
