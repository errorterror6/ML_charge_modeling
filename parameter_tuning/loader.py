import pickle, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os, json

from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

import torch, math

import parameters
import init
import visualisation

import sys
sys.path.append('../libs/')
import shjnn

'''
@title: load_data
@description: loads data from raw files into the specified dataset struct. typically parameters.dataset (in parameters.py)
'''
def load_data(data_out=parameters.dataset):

    ## load raw data from file

    # open binary file for reading
    with open('{}/{}'.format('../run/', 'raw-full'), 'rb') as file:

        # load pickled data storage array
        db = pickle.load(file)


    ## parameter mappings to dimension data

    val_map = {
        # sample id to thickness in nm
        'thickness': {'source': 'sample',
                    '12': 160, '16': 160, '21': 160, '33': 160, '45': 160, '58': 160, },
        # intensity in uJ
        'intensity': {'source': 'int',
                    'dark': 0., '32uJ': 32., '10uJ': 10., '3uJ': 3., '1uJ': 1., '03uJ': .3},
        # voltage in V
        'voltage': {'source': 'vlt',
                    '05V': .5, '0V': 0., '15V': 1.5, '1V': 1., '2V': 2.},
        # delay time in log10(s)
        'delay': {'source': 'del',
                '100ns': 1e-7, '100us': 1e-4, '10ms': 1e-2, '10us': 1e-5, '1ms': 1e-3, '1us': 1e-6,
                '200ns': 2e-7, '200us': 2e-4, '20ms': 2e-2, '20us': 2e-5, '2ms': 2e-3, '2us': 2e-6,
                '500ns': 5e-7, '500us': 5e-4, '50ms': 5e-2, '50us': 5e-5, '5ms': 5e-3, '5us': 5e-6,
                },
    }

    ''' generate properties data '''

    # iterate nodes in database
    for node in db[:]:
        
        props = {}
        
        # iterate value mapping
        for key, value in val_map.items():
        
            # check node params for value
            if value['source'] in node['params'].keys():
                
                # map param to prop value
                props[key] = value[ node['params'][ value['source'] ] ]
                            
            else:
                # store zero for no match
                props[key] = 0.
                
        # store props in node
        node['props'] = props
    ''' generate properties data '''

    # iterate nodes in database
    for node in db[:]:
        
        props = {}
        
        # iterate value mapping
        for key, value in val_map.items():
        
            # check node params for value
            if value['source'] in node['params'].keys():
                
                # map param to prop value
                props[key] = value[ node['params'][ value['source'] ] ]
                            
            else:
                # store zero for no match
                props[key] = 0.
                
        # store props in node
        node['props'] = props

    # ''' compile lists params '''

    # compile list samples
    smpls = sorted(list(set([ _['params']['sample'] for _ in db ])))
    print( 'smpls', smpls )

    # compile list voltages
    vlts = sorted(list(set([ _['props']['voltage'] for _ in db ])))
    print( 'vlts', vlts )

    # compile list voltages
    ints = sorted(list(set([ _['props']['intensity'] for _ in db ])))
    print( 'ints', ints )

    # complete list delays
    #dels = sorted(list(set([ _['props']['delay'] for _ in db ])))[1::]
    # restricted delay set
    dels = sorted(list(set([ _['props']['delay'] for _ in db ])))[1::3]
    print( 'dels', dels )

    #  ''' get full single-sample dataset '''

    # select transient conditions
    _smpl = '33'
    #_int = '10uJ'
    #_vlt = '15V'
    #_del = '10us'

    _db = [ n for n in db if
        n['params']['sample'] == _smpl and
        n['props']['delay'] != 0.
        #n['params']['int'] == _int and
        #n['params']['vlt'] == _vlt and
        #n['params']['del'] == _del
        ]

    print(len(_db))


    # initialise figure
    fig = plt.figure(figsize = (8,6))
    #axs = [[ fig.add_subplot(v,h,i+1) for i in range(h*v) ] for h,v in [(2,1)]][0]
    axs = [[ fig.add_subplot(v,h,i+1) for i in range(h*v) ] for h,v in [(1,1)]][0]

    # colourmap
    cnorm  = colors.Normalize(vmin = 0, vmax = len(_db[::10])); smap = cmx.ScalarMappable(norm = cnorm, cmap = 'viridis')

    # zero baseline
    axs[0].plot([1e-8, 1e-4], [0,0], '--k')


    for i, n in enumerate(_db[::10]):
        
        # get time axis
        t = n['data'][:,0]
        d = n['data'][:,1]/50.
        #d0 = dark0['data'][:,1]#/50.
        #ds = d-d0

        c = smap.to_rgba(i)

        #print(light['params']['del'])

        #np.power(10,val_map['delay'][light['params']['del']])
        #sub = np.vstack([dark[:,0], (light[:,1] - dark[:,1])/50. ]).T

        #l = light['data'][:,1]#/50.
        #s = l-d

        # calculate current transient, illuminated less dark transient, voltage over 50 Ohm res
        #a = (light[:,1] - dark[:,1])/50.

        # smooth and calculate derivatives
        a = savgol_filter(x = d, window_length = 5, polyorder = 1, deriv = 0)

        _ds = 10
        
        # plot data
        axs[0].plot(t[::_ds], d[::_ds], '.', alpha = 0.2, color = c, linewidth = 2)
        axs[0].plot(t[::_ds], a[::_ds], '-', alpha = 0.44, color = c, linewidth = 1, label = ''.format())


        #axs[1].plot(t, da, '-', alpha = 0.7, color = c, linewidth = 1, label = '{}'.format(light['params']['int']))

            
    # format plots
    for i in range(len(axs)):
        axs[i].set_xscale('log')
        axs[i].set_xlim(1.0e-8, 1.0e-4)
        #axs[i].set_xlim(2.0e-7, 1.0e-4)
        
        #axs[i].set_ylim(0., 1.0e-2)
        axs[i].set_xlabel('Extraction Time (s)')
        axs[i].set_ylabel('Extracted Charge (A)')
        #axs[i].legend(title='Illumination')
        
    plt.tight_layout()
    # plt.show()

    if filter:
        # filter data to only include non-zero delay times
        j = np.where(_db[0]['data'][:,0] > 2e-7)[0]

        # get log time steps, shift decay start to zero
        d = np.transpose( np.stack([ [ 
        # np.log10(n['data'][j,0][::_ds]), # time

        n['data'][j,1][::_ds]*1e2, # response, scale to ~one
        #np.log10(n['data'][j,0][::_ds])+6.6,

        #np.ones(len(ts))* np.log10(n['props']['intensity']),
        #np.ones(len(ts))* n['props']['voltage']/10,
        #np.ones(len(ts))* np.abs(np.log10(n['props']['delay']))/10,
        ]
        #for n in _db ]), (2,0,1))
        for n in _db ]), (0,2,1))
        
        # compile env. labels
        ts = np.transpose( np.stack([ [ 
            #n['data'][:,0][::_ds], # time
            #n['data'][j,0][::_ds]
            
            np.log10(n['data'][j,0][::_ds])+6.7,
        ]
        #for n in _db ]), (2,0,1))
            for n in _db ]), (0,2,1))

    else:
        j = np.where(_db[0]['data'][:,0] > 1e-9)[0]
        # get log time steps, shift decay start to zero
        #ts = np.log10(db[0]['data'][:,0][::_ds])
        ts = np.transpose( np.stack([ [ 
            #n['data'][:,0][::_ds], # time
            #n['data'][j,0][::_ds]
            
            np.log10(n['data'][j,0][::_ds])+6.7,
        ]
        #for n in _db ]), (2,0,1))
            for n in _db ]), (0,2,1))


        # stack raw data and env vars, time first
        #d = np.transpose( np.stack([ [ n['data'][:,1][::_ds], 
        d = np.transpose( np.stack([ [ 
            #np.log10(n['data'][j,0][::_ds]), # time
            
            n['data'][j,1][::_ds]*1e2, # response, scale to ~one
            #np.log10(n['data'][j,0][::_ds])+6.6,
            
        #np.ones(len(ts))* np.log10(n['props']['intensity']),
        #np.ones(len(ts))* n['props']['voltage']/10,
        #np.ones(len(ts))* np.abs(np.log10(n['props']['delay']))/10,
        ]
        #for n in _db ]), (2,0,1))
            for n in _db ]), (0,2,1))

    print("Logs: Loader: load_data: ts.shape, d.shape: ", ts.shape, d.shape)

    ''' compile env. labels '''

    # stack raw data and env vars, time first
    y = np.stack([ [ 
    n['props']['intensity'],
    n['props']['voltage'],
    n['props']['delay'],
    n['props']['thickness'],
    ]
        for n in _db ])

    print("Logs: loader: load_data, stack shape", y.shape)

    # compile trajectories, time as tensors, push to device
    trajs = torch.Tensor(d).to(parameters.device)
    times = torch.Tensor(ts).to(parameters.device)

    print("Logs: loader: load_data, length of trajs, shape of trajs[0], shape of times[0]: ", len(d), d[0].shape, ts[0].shape)

    parameters.dataset['trajs'] = trajs
    parameters.dataset['times'] = times
    parameters.dataset['y'] = y

def save_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, random_samples=True):
    visualisation.display_random_fit(model_params, dataset, show=False, save=True, random_samples=random_samples)

def save_model_params(model_params=parameters.model_params):
    #save model params as json file
    folder = model_params['folder']
    epoch = model_params['epochs']
    model_params_save = {
        # hyper params

        
        'nhidden': model_params['nhidden'],
        'rnn_nhidden': model_params['rnn_nhidden'],
        'obs_dim': model_params['obs_dim'],

        'latent_dim': model_params['latent_dim'],
        'lr': model_params['lr'],
        'n_batch': model_params['n_batch'],
        'beta': model_params['beta'],

        # training params
        'total_epochs_train': model_params['total_epochs_train'],
        'epochs_per_train': model_params['epochs_per_train'],
        'epochs': model_params['epochs'], # a record of the epochs
        'loss': model_params['loss'], 
        'MSE_loss': model_params['MSE_loss'],
        'KL_loss': model_params['KL_loss'],

        #labels
        'name': model_params['name'],
        'desc': model_params['desc'],
        'folder': model_params['folder'],
    }

    if not os.path.exists(folder + '/model'):
        os.makedirs(folder + '/model')
    with open(folder + f'/model/model_params_epoch_{epoch}.json', 'w') as fp:
        json.dump(model_params_save, fp)

def load_model_params():
    #load model params from json file
    folder = model_params['folder']
    epoch = 5   #enter your desired epoch to extract here
    with open(folder + f'/model/model_params_{epoch}.json', 'r') as fp:
        model_params = json.load(fp)
        # init.init_model(model_params)
    return model_params

def clear_saves():
    ''' clear all save files in saves directory '''
    import os
    import glob
    #todo change??
    files = glob.glob('saves/*')
    for f in files:
        os.chmod(f, 0o777)
        os.remove(f)

def check_environment():
    import numpy as np
    import pandas as pd

    from scipy.interpolate import splrep, splev
    from scipy.signal import savgol_filter
    import os, re, logging

    # %matplotlib widget
    #import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize

    import pickle, random

    ''' torch imports '''

    import torch, math

    from torch import nn
    from torch import optim
    from torch.distributions import Normal

    import glob
    from PIL import Image
    import matplotlib.animation as animation

    import json

    #import torchsde, torchcde



    ''' init shjnn dev env '''

    import os
    # create save and gif folders if they dont exist already
    if not os.path.exists('./saves/'):
        os.makedirs('./saves/')
    if not os.path.exists('./gifs/'):
        os.makedirs('./gifs/')

    # add location to path
    import sys
    sys.path.append('../libs/')

    import shjnn
    import time
    import ffmpeg

def save_model(model_params):
    parameters.model.save_model(model_params)

def load_model(model_params):
    #load model params from json file. Overwrite model params.
    folder = ""
    epoch = 5   #enter your desired epoch to extract here
    path = folder + f'/model/save_model_ckpt_{epoch}.pth'
    try:
        load_model_params()
    except:
        print("Logs: Loader: load_model: no model params found, loading model without model params.")
    parameters.model.load_model(model_params, path)
