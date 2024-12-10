import os, re, logging, torch, random, savgol_filter, glob
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation

import parameters
import sys
sys.path.append('../libs/')
import shjnn

def plot_training_loss(model_params, save=False, split=False, plot_total=False, plot_MSE=True, plot_KL=True):
    ''' plot training loss '''
    if split:
        MSE_loss = model_params['MSE_loss']
        KL_loss = model_params['KL_loss']
        total_loss = model_params['loss']

        k = 1
        _w = 5
        _h = 2*k
        fig = plt.figure(figsize=(_w, _h))

        # Simplified subplot addition
        ax = fig.add_subplot(1, 1, 1)

        # Plot MSE_loss with red color on the primary y-axis
        if plot_MSE:
            ax.plot(MSE_loss, '-r', label='MSE loss', alpha=0.3, color='red')
            ax.set_yscale('log')
            ax.set_ylabel('MSE Loss')

        # Plot KL_loss with blue color on the secondary y-axis
        if plot_total:
            ax2 = ax.twinx()
            ax2.plot(total_loss, '-b', label='total loss', alpha=0.3, color='blue')
            ax2.set_yscale('log')
            ax2.set_ylabel('Total Loss')

        # add the KL loss to another axis
        if plot_KL:
            ax1 = ax.twinx()
            ax1.plot(KL_loss, '-g', label='KL loss', alpha=0.3, color='green')
            # ax1.plot(-np.array(MSE_loss) + np.array(total_loss), '-g', label='KL loss2', alpha=0.3, color='green')
            ax1.set_ylabel('KL Loss')

        # Smooth loss using Savitzky-Golay filter for both MSE_loss and KL_loss
        # train_MSE_loss = np.abs(savgol_filter(MSE_loss, 13, 3))
        # train_KL_loss = np.abs(savgol_filter(KL_loss, 13, 3))

        # Optionally plot smoothed losses (uncomment to use)
        # ax.plot(train_MSE_loss, '-r', alpha=0.8)
        # ax1.plot(train_KL_loss, '-b', alpha=0.8)

        # Labels and legend
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Losses')

        # Only call plt.legend() after setting labels for each line plot.
        ax.legend(loc='upper left')
        if plot_KL:
            ax1.legend(loc='upper right')
        if plot_total:
            ax2.legend(loc='upper center')
        
        # add the final loss value into the title
        plt.title(f'Final Loss: {round(total_loss[-1])}')
        plt.tight_layout()
        plt.show()

        if save:
            folder = model_params['folder']
            save_folder = f"{folder}/loss_graph/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(f"{save_folder}/loss_graph_split.png")
    else:
        # get loss
        loss = model_params['loss']

        # initialise figure
        k = 1; _w = 5; _h = 2*k; fig = plt.figure(figsize = (_w, _h))
        #fig.canvas.layout.width = '{}in'.format(_w); fig.canvas.layout.height= '{}in'.format(_h)
        ax = [ [ fig.add_subplot(j,1,i) for i in range(1,j+1) ] for j in [k] ][0]


        # plot original and predicted trajectories
        ax[0].plot([_+1e1 for _ in loss], '-b', label = 'training loss', alpha = 0.3)

        # smooth loss
        train_loss = np.abs( savgol_filter(loss, 13, 3) )

        # plot original and predicted trajectories
        ax[0].plot([_+1e1 for _ in train_loss], '-b', alpha = 0.8)


        # format and display figure
        plt.yscale('log')
        #plt.xscale('log')

        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.legend()
        plt.tight_layout()
        plt.show()

        if save:
            folder = model_params['folder']
            save_folder = f"{folder}/loss_graph/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(f"{save_folder}/loss_graph.png")

def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
    ''' random assess model fit '''


    # get data
    trajs = dataset['trajs']
    times = dataset['times']
    y = dataset['y']

    # get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    optim = model_params['optim']
    device = model_params['device']
    epoch = model_params['epochs']
    

    # initialise figure
    k = trajs[0].shape[-1]; _w = 7; _h = 4*k; fig = plt.figure(figsize = (_w, _h))
    #fig.canvas.layout.width = '{}in'.format(_w); fig.canvas.layout.height= '{}in'.format(_h)
    ax = [ [ fig.add_subplot(j,1,i) for i in range(1,j+1) ] for j in [k] ][0]

    # generate inference function
    infer_step = shjnn.make_infer_step(func, rec, dec, optim, device, _input = 'traj', _sample=False)

    # select data
    j = list(range(len(trajs)))
    if random_samples:
        random.shuffle(j)

    # downsample
    j = j[::30]

    # build colourmap
    cnorm  = colors.Normalize(vmin = 0, vmax = len(j)); smap = cmx.ScalarMappable(norm = cnorm, cmap = 'brg')

    # iterate over transients
    for _,i in enumerate(j):
        
        # get colour
        c = smap.to_rgba(_)
        
        # send mini-batch to device
        traj = trajs[i].view(1, *trajs[i].size()).to(device)
        
        #_time = np.linspace(-7.8, -4.2, 1000)#/10
        #_time = np.linspace(-6.5+6.6, -4.2+6.6, 1000)#/10
        
        _time = np.linspace(0, 2.5, 1000)#/10
        
        #_time = np.linspace(-7., -4.2, 1000)
        #_time = np.logspace(-7.8, -4.2, 1000)
        
        #_time = np.logspace(0, 1.7, 20)
        time = torch.Tensor(_time).to(device)

        # perform inference step for prediciton
        pred_x, pred_z = infer_step(traj, time)

        # return prediction to cpu
        pred_x = pred_x.cpu().numpy()[0]
        
        #print(pred_x.shape, pred_z[0,0,:])
        
        _traj = trajs[i].cpu()
        _t = times[i].cpu()

        for l in range(k):
            u = 0
            
            #ax[k].set_ylim(-.8, .8)
            sc_ = 50*1e2/1e3
            
            # plot original and predicted trajectories
            ax[l].plot(_t, _traj[:, l+u]/sc_, '.', alpha = 0.6, color = c)
            ax[l].plot(_time - 1.0, pred_x[:, l+u]/sc_, '-', label = '{:.1f} $\mu J$, {:.1f} V, {:.0e} s'.format(y[i][0], y[i][1], y[i][2]),
                    linewidth = 2, alpha = 0.4, color = c)

            
    plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
    plt.ylabel('Charge [mA]')
    # tile includes epoch number, learning rate atnd beta
    plt.title('Epoch: {}, lr: {:.1e}, beta: {:.1e}'.format(epoch, model_params['lr'], model_params['beta']))

    #plt.xscale('log')
    plt.legend(loc='upper right', title='Intensity, Bias, Delay')

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        #save as a png]
        #todo: change
        folder = model_params['folder']
        #if saves folder does not exist create it
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(folder + '/training_epoch_{}.png'.format(epoch), dpi=300)

def compile_learning_gif(model_params=parameters.model_params, display=True):
    # compile gif from png in saves folder
    # set read destinatino here
    folder = model_params['folder']
    glob_target = folder + '/*.png'
    print(glob_target)
    files = sorted(glob.glob(glob_target))
    image_array = []

    for my_file in files:
        
        image = Image.open(my_file)
        image_array.append(image)

    print('image_arrays shape:', np.array(image_array).shape)

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)

    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=200, blit=True,repeat_delay=10,)

    # Show the animation
    plt.show()

    # animation_fig.save("gifs/animated_GMM.gif")

    
    # to html5 video using pillow
    if not os.path.exists(folder + "/gifs/"):
        os.makedirs(folder + "/gifs/")
    animation_fig.save(folder + "/gifs/learning_pattern.gif", writer="pillow")

    # if (display):
    #     video = animation_fig.to_html5_video()
    #     html = display.HTML(video)
    #     display.display(html) 

def sweep_latent_adaptives(model_params, dataset):
    for j in range(model_params['latent_dim']):
        sweep_latent_adaptive(model_params, dataset, j)

def sweep_latent_adaptive(model_params, dataset, latent_dim_number):
    ''' get z0 prediction of complete dataset '''

    # get data
    trajs = dataset['trajs']
    times = dataset['times']
    y = dataset['y']

    #get model
    func = model_params['func']
    rec = model_params['rec']
    dec = model_params['dec']
    optim = model_params['optim']
    device = model_params['device']
    epoch = model_params['epochs']
    latent_dims = model_params['latent_dim']

    # generate inference function
    infer_step = shjnn.make_infer_step(func, rec, dec, optim, device, _input = 'traj', _sample=False)

    # select data
    j = list(range(len(trajs)))
    #random.shuffle(j)

    Z = []

    Zz = []

    for i in j[::]:
        
        # send mini-batch to device
        traj = trajs[i].view(1, *trajs[i].size()).to(device)
        
        _time = np.linspace(0, 2.5, 1000)#/10
        
        #_time = np.linspace(-7., -4.2, 1000)
        #_time = np.logspace(-7.8, -4.2, 1000)
        
        #_time = np.logspace(0, 1.7, 20)
        time = torch.Tensor(_time).to(device)

        # perform inference step for prediciton
        pred_x, pred_z = infer_step(traj, time)
        
        Z.append(pred_z[0, 0, ...].detach().numpy())
        
        Zz.append(pred_z[0, ...].detach().numpy())

        # return prediction to cpu
        pred_x = pred_x.cpu().numpy()[0]
        
    Z = np.stack(Z)
    Zz = np.stack(Zz)

    print(Z.shape, Zz.shape)

    ''' sweep latent adaptives '''

    # initialise figure
    k = trajs[0].shape[-1]; _w = 7; _h = 4*k; fig = plt.figure(figsize = (_w, _h))
    #fig.canvas.layout.width = '{}in'.format(_w); fig.canvas.layout.height= '{}in'.format(_h)
    ax = [ [ fig.add_subplot(j,1,i) for i in range(1,j+1) ] for j in [k] ][0]
    # setup ax for each latent dim


    # generate inference function
    #infer_step = shjnn.make_infer_step(func, rec, dec, optim, device, _input = 'traj', _sample=False)
    infer_step = shjnn.make_infer_step(func, rec, dec, optim, device, _input = 'latent')

    # set z dim to sweep
    j = latent_dim_number
    # for j in range(latent_dims):

    # set range over latent vector
    rr = 3
    _ = np.linspace(-rr,rr,10)

    # colourmap
    cnorm  = colors.Normalize(vmin = 0, vmax = len(_)); smap = cmx.ScalarMappable(norm = cnorm, cmap = 'cividis')

    # iterate over latent vector range
    for i in range(len(_)):
        _z = _[i]
        c = smap.to_rgba(i)

        # set init latent to mean of dataset or zeros
        _z0 = np.expand_dims(np.mean(Z, 0),0)
        #_z0 = np.expand_dims(np.zeros(Z.shape[-1]),0)
        
        # update latent vector for variation
        _z0[...,j] += _z
        #print(_z0)
        
        z0 = torch.Tensor(_z0).to(device)

        # define time axis
        _time = np.linspace(0, 2.5, 1000)#/10
        time = torch.Tensor(_time).to(device)

        # perform inference step for prediciton
        pred_x, pred_z = infer_step(z0, time)

        # return prediction to cpu
        pred_x = pred_x.cpu().numpy()[0]

        # plot predicted trajectories
        ax[0].plot(_time, pred_x[:, 0], '-', label = 'z{}, $\mu$ {:.1f} + {:.1f}'.format(j, np.mean(Z, 0)[j],_z), alpha = 0.6, color = c, linewidth = 2)

            
    plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
    plt.ylabel('Charge [mA]')

    plt.hlines(0., -.1, 2.6, colors = 'k', linestyle = '--', alpha = 0.5)
    plt.xlim(-.1,2.6)
            
    plt.legend()
        
    #plt.xscale('log')
    plt.tight_layout()
    plt.show()

    folder = model_params['folder']
    #if saves folder does not exist create it
    if not os.path.exists(folder + '/latent_dims'):
        os.makedirs(folder + '/latent_dims')
    fig.savefig(folder + f'/latent_dims/epoch_{epoch}_dim_{latent_dim_number}.png', dpi=300)
