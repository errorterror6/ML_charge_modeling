
''' imports '''

import numpy as np

import torch

# ode solve component
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint



''' training orchestration '''

def make_train_step(func, rec, dec, optim, device, noise_std = 0.3, beta = 1.0):

    ''' Build Training Step Function

        builds function that performs a step in the train loop

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    def train_step(traj, time):

        # set models to train mode
        rec.train()
        func.train()
        dec.train()


        ''' ingest trajectory (reverse time) '''

        # initialise recognition network hidden layer
        h = rec.initHidden().to(device)

        # iterate length each trajectory input
        #for t in reversed(range(traj.size(0))):
        for t in reversed(range(traj.size(1))):

            # get trajectory samples in reverse time from final
            #obs = x[t, :].view(1,-1)
            obs = traj[:, t, :]

            # run trajectory through recog net
            out, h = rec.forward(obs, h)


        ''' infer initial latent state '''

        latent_dim = out.size()[1]//2

        # split final recog net output, latent state mean and variance
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        #qz0_mean, qz0_logvar = out[:, :latent_dim].squeeze(), out[:, latent_dim:].squeeze()

        # generate random tensors size latent space dims (sample random normal distribution)
        epsilon = torch.randn(qz0_mean.size()).to(device)

        # infer initial augmented state from posterior (random sample)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        #print('z0 ', z0.size())

        ''' compute state predictions '''

        #print('time ', time.size())
        #print('time 2 ', time.squeeze().size())

        atol = 1e-6
        # get state prediction from ode solver given inferred state and samples
        pred_z = odeint(func, z0, time.squeeze(), atol = atol).permute(1, 0, 2)

        #print('z ', pred_z.size())


        # decode predicted state (latent) to final trajectory dimensions
        pred_x = dec(pred_z)

        #print('x ', pred_x.size(), '\n')


        ''' compute loss '''

        # generate noise, size of prediction
        noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)

        # log-normal probability density function over samples, prediction with noise
        logpx = log_normal_pdf(traj, pred_x, noise_logvar).squeeze().sum(-1).sum(-1)


        # initialse mean, var for initial state predictions as zeros
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)

        # calculate cross-entropy loss (kl-divergence)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)


        # calculate total loss
        loss = -logpx + beta * analytic_kl

        #print(loss)

        ''' perform backprop, param update '''

        # compute gradients
        loss.backward()

        # update parameters
        optim.step()

        # zeroes gradients
        optim.zero_grad()


        # returns calculated loss
        return loss.item()


    # return function to be called within train loop
    return train_step





''' training components '''

def log_normal_pdf(x, mean, logvar):

    ''' log-normal probability density function

        compute log-normal pdf
    '''

    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)

    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))



def normal_kl(mu1, lv1, mu2, lv2):

    ''' kl-divergence

        compute kl-divergence for cross-entropy
    '''
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)

    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5

    return kl
