import parameters
from torch import nn
import visualisation

import sys
sys.path.append('../libs/')
import shjnn

class B_VAE:
    class visualiser:
        def __init__(self):
            pass

        def plot_training_loss(model_params=parameters.model_params, save=False, split=False, plot_total=False, plot_MSE=True, plot_KL=True):
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL)

        def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
            visualisation.display_random_fit(model_params, dataset, show=show, save=save, random_samples=random_samples)
        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)



            
            
