import parameters
from torch import nn
import torch, random, glob, os
import visualisation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx


import sys
sys.path.append('../libs/')
import shjnn

class RNN:
    model = None
    criterion = None
    model_params = None

    def __init__(self, model, model_params=parameters.model_params):
        RNN.model = model
        RNN.criterion = nn.MSELoss()
        RNN.model_params = model_params

   
        

    def train_step(traj, time):
        RNN.model.train()
        h = RNN.model.initHidden().to(RNN.model_params['device'])

        # iterate length each trajectory input
        #for t in reversed(range(traj.size(0))):
        # TODO： put the time step into the RNN, unreverse this.
        for t in range(traj.size(1)):

            # get trajectory samples in reverse time from final
            #obs = x[t, :].view(1,-1)
            obs = traj[:, t, :]
            # print("obs shape:;" , obs.shape, obs)

            # run trajectory through recog net
            out, h = RNN.model.forward(obs, h)
        loss = RNN.criterion(out, obs)
        return loss
    
    def train(n_epochs, model_params=parameters.model_params, dataset=parameters.dataset):

        dataset = shjnn.CustomDataset(dataset['trajs'], dataset['times'])
        # TODO: TRAJS should have all of the experimental variables (5 of em)
        # print("Logs: rnn: train: printing trajectories size: ", dataset['trajs'][0].numpy())

        # split dataset into training, validation
        #train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [90, 10])
        train_dataset = dataset

        # initialise training data loader for random mini-batches
        #TODO: look at shjnn.dataLoader more.
        train_loader = shjnn.DataLoader(train_dataset, batch_size = model_params['n_batch'], shuffle = True, drop_last = True)
        n = len(train_dataset)

        loss = []
        print("Logs: rnn: train: printing n_epochs: ", n_epochs)
        for epoch in range(1, n_epochs+1):
            try:
                for x_batch, y_batch in train_loader:
                    _loss = RNN.train_step(x_batch, y_batch)
                    loss.append(_loss.detach().numpy())
            except KeyboardInterrupt:
                return epoch, loss
        epoch_loss = sum(loss) / len(loss)
        return epoch, loss
    
    class visualiser:
        def __init__(self):
            pass

        def plot_training_loss(model_params=parameters.model_params, save=True, split=False, plot_total=True, plot_MSE=False, plot_KL=False):
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=plot_MSE, plot_KL=plot_KL)

        def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
            ''' random assess model fit '''
            print("Logs: rnn: visualiser: display_random_fit: ")

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
            def infer_step(traj, time):
                # inference, do not compute gradients
                with torch.no_grad():

                    # set models to evaluation only mode
                    rec.eval()
                    pred_x = rec(traj)
                    


                # returns calculated loss
                pred_z = []
                return pred_x, pred_z

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
                
                #+1 to account for time bias associated with removing the initial rise.
                _time = np.linspace(0, 2.5, 1000) + 1#/10
                
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
                    ax[l].plot(_time - 1.0, pred_x[:, l+u]/sc_, '-', label = '{:.1f} J$, {:.1f} V, {:.0e} s'.format(y[i][0], y[i][1], y[i][2]),
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
                fig.savefig(f"{folder}/training_epoch_{epoch:04d}.png", dpi=300)

        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)

