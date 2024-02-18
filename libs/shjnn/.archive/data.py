
''' imports '''

import pickle, os

import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn import preprocessing



''' import and prepare dataset '''

def prep_data():

    ''' Prepare Dataset

        pre-process data, prepare for training

    Args:
        var (int): some variable

    Returns:
        (str): some output
    '''

    ''' import raw dataset '''

    # dataset location
    _base_path, _file_name = '../data', 'db'

    # open dataset file
    with open(os.path.join(_base_path, _file_name), 'rb') as file:

        # load pickled data dict
        db = pickle.load(file)


    ''' filter dataset '''

    # get list unique devices by id
    #devs = list(set([ n['device'] for n in db if int(n['device']) in range(16, 21) ]))
    devs = list(set([ n['device'] for n in db
                     #if n['intensity'] in [1550, 740, 400]
                     #and n['temperature'] in [180, 150, 120]

                     if n['intensity'] not in [1550, 740]
                     and n['temperature'] not in [180, 25]

                     #and n['proc_time'] < 100
                     ]))


    ''' aggregate data into list features, index '''

    # store device data in list
    data = []
    time = []

    # define data to extract by key
    #keys = ['temperature', 'intensity', 'proc_time', 'voc', 'ff', 'rs']
    keys = ['temperature', 'intensity', 'voc', 'ff', 'rs']
    #keys = ['voc', 'ff', 'rs']

    dep_var = 'proc_time'

    # iterate each device
    for dev in devs[:]:

        # find measurement nodes by device id
        nodes = [ n for n in db if n['device'] == dev ]

        # sort nodes on processing dependent variable (time axis)
        j = np.argsort( np.array([ n[dep_var] for n in nodes ]) )
        nodes = [ nodes[i] for i in j ]

        # extract data from each node and store
        _data = np.array([ [ node[k] for k in keys ] for node in nodes[:] ])

        #_time = np.array([ [ node[k] for k in [dep_var] ] for node in nodes[:] ])
        _time = np.array([ node[dep_var] for node in nodes[:] ])

        # store data array
        data.append( _data )
        time.append( _time )


    ''' normalise dataset features '''

    # initialise scaler for data, fit to all data (flatten on time axis)
    data_scaler = preprocessing.StandardScaler().fit( np.concatenate(data) )

    # scale / normalise data
    norm_data = [ data_scaler.transform(_) for _ in data ]


    ''' convert to torch tensors '''

    # prepare dataset (trajectory vector and time) as tensors, move to device
    #trajs = [ torch.Tensor(_).to(device) for _ in norm_data ]
    #times = [ torch.Tensor(_).to(device) for _ in time ]
    trajs = [ torch.Tensor(_) for _ in norm_data ]
    times = [ torch.Tensor(_) for _ in time ]


    # return data lists: trajectories, times; data scaler
    return trajs, times, data_scaler



''' dataset components '''

class CustomDataset(Dataset):

    def __init__(self, x_tensor, y_tensor):

        self.x = x_tensor
        self.y = y_tensor


    def __getitem__(self, index):

        return (self.x[index], self.y[index])


    def __len__(self):

        return len(self.x)


