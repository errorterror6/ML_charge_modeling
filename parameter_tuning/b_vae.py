import parameters
import torch
from torch import nn
import visualisation
import numpy as np
import loader
import os

import sys
sys.path.append('../libs/')
import shjnn

class B_VAE:
    def __init__(self, m=parameters.model_params):
        self.visualiser = self.Visualiser(self)

    def eval(self, model_params=parameters.model_params, dataset=parameters.dataset):
        #in visualisation.py, traj_tensor is of size 70 and if you feed the infer method with a time tensor that's unextrapolated (70),
        #it will return a tensor of size 70 which is the prediction and now you can MSE it with the original time_tensory.
        # Extract data from dataset - use original data for evaluation
        trajectories = dataset['trajs']  # Original trajectories without missing data
        time_points = dataset['times']   # Original time points without missing data
        metadata = dataset['y']  # Contains parameters like intensity, bias, delay

        # Extract model components
        model_func = model_params['func']
        encoder = model_params['rec']
        decoder = model_params['dec']
        optimizer = model_params['optim']
        device = model_params['device']
        epoch_num = model_params['epochs']
        
        num_dims = trajectories[0].shape[-1]
        
        infer_step = shjnn.make_infer_step(
            model_func, encoder, decoder, optimizer, device, 
            input_mode='traj', sample=False
        )
        sample_indices = list(range(len(trajectories)))
        print(f"logs: b_vae: eval: evaluating over {len(sample_indices)} samples.")
        loss_list = []
        for idx, traj_idx in enumerate(sample_indices):
            # Free up GPU memory between samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traj_tensor = trajectories[traj_idx].view(1, *trajectories[traj_idx].size()).to(device)
            time_tensor = time_points[traj_idx].view(1, *time_points[traj_idx].size()).to(device)
            
            pred_x, pred_z = infer_step(traj_tensor, time_tensor)
            
            #currently both at [1, 70, 1]
            # No need for redundant .to(device) calls - tensors are already on the device
            loss = torch.nn.MSELoss()(pred_x, traj_tensor)
            loss_list.append(loss.item())
        print(f"logs: b_vae: eval: mean loss over {len(sample_indices)} samples: {np.mean(loss_list)} at epoch {epoch_num}.") 
        return np.mean(loss_list)
            
    def save_model(self, model_params):
        #save model params as json file
        folder = model_params['folder']
        epoch = model_params['epochs']
        if not os.path.exists(folder + '/model'):
            os.makedirs(folder + '/model')

        # save model
        #load model params
        func = model_params['func']
        rec = model_params['rec']
        dec = model_params['dec']
        optim = model_params['optim']
        loss = model_params['loss']
        epochs = model_params['epochs']
        folder = model_params['folder']
        path = folder + f'/model/save_model_ckpt_{epoch}.pth'
        shjnn.save_state(path, func, rec, dec, optim, loss, epochs) 
        
        loader.save_model_params(model_params)
        
    def load_model(self, model_params, path):
        shjnn.load_state(path, model_params['func'], model_params['rec'], dec = model_params['dec'], optim = model_params['optim'], loss = model_params['loss'], epochs = model_params['epochs'], dev = model_params['device'])


    class Visualiser:
        def __init__(self, b_vae_instance):
            pass


        def plot_training_loss(self, model_params=parameters.model_params, save=False, split=False, plot_total=True, plot_MSE=True, plot_KL=True):
            # NOTE: uncomment or comment these to toggle between.
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=False, plot_KL=False)
            # visualisation.plot_training_loss(model_params, save=save, split=True, plot_total=False, plot_MSE=plot_MSE, plot_KL=plot_KL)
            

        def display_random_fit(self, model_params=parameters.model_params, dataset=parameters.dataset, show=False, save=True, random_samples=True):
            visualisation.display_random_fit(model_params, dataset, show=show, save=save, random_samples=random_samples)
        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(self, model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(self, model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)
