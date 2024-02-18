
''' imports '''

import numpy as np

import torch

# ode solve component
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint



''' model inference '''

def make_infer_step(func, rec, dec, optim, device, _input = 'traj', _sample=True):

    ''' Build Inference Step Function

        builds function that performs an inference step

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    # if passed latent dimensions directly
    if _input == 'latent':

        def infer_step(z0, time):

            # inference, do not compute gradients
            with torch.no_grad():

                # set models to evaluation only mode
                rec.eval()
                func.eval()
                dec.eval()

                ''' compute state predictions '''

                #print(time.size())

                # get state prediction from ode solver given inferred state and samples
                pred_z = odeint(func, z0, time.squeeze()).permute(1, 0, 2)

                #print(pred_z.size())
                #print(pred_z)

                # decode predicted state (latent) to final trajectory dimensions
                pred_x = dec(pred_z)


            # returns calculated loss
            return pred_x, pred_z

    # input as full trajectories
    else:

        def infer_step(traj, time):

            # inference, do not compute gradients
            with torch.no_grad():

                # set models to evaluation only mode
                rec.eval()
                func.eval()
                dec.eval()


                ''' ingest trajectory (reverse time) '''

                # initialise recognition network hidden layer
                h = rec.initHidden().to(device)[:1,:]
                #print(h.shape)

                # iterate length each trajectory input
                for t in reversed(range(traj.size(1))):

                    # get trajectory samples in reverse time from final
                    obs = traj[:, t, :]

                    # run trajectory through recog net
                    out, h = rec.forward(obs, h)


                ''' infer initial latent state '''

                # predicted latent state mean
                latent_dim = out.size()[1]//2

                # split final recog net output, latent state mean and variance
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                #qz0_mean, qz0_logvar = out[:, :latent_dim].squeeze(), out[:, latent_dim:].squeeze()

                # generate random tensors size latent space dims (sample random normal distribution)
                epsilon = torch.randn(qz0_mean.size()).to(device)

                # infer initial augmented state from posterior (random sample)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # if not sampling, use direct z0 prediction
                if _sample == False:
                        z0 = qz0_mean

                #z0 = out[:, :latent_dim]

                #print(z0.size())


                ''' compute state predictions '''

                #print(time.size())

                # get state prediction from ode solver given inferred state and samples
                pred_z = odeint(func, z0, time.squeeze()).permute(1, 0, 2)
                #pred_z = odeint(func, z0, time[0,:]).permute(1, 0, 2)

                #print(pred_z.size())
                #print(pred_z)

                # decode predicted state (latent) to final trajectory dimensions
                pred_x = dec(pred_z)


            # returns calculated loss
            return pred_x, pred_z


    # return function to be called within train loop
    return infer_step
