import numpy as np
import torch

import parameters

def missing_data_single(data, random_rate=0, drop_array=None):
    """
    Create missing data in the input data.
    
    Args:
        data (np.ndarray): Input data.
        missing_rate (float): Rate of missing data.
    
    Returns:
        np.ndarray: Data with missing values.
    """
    missing_data = np.copy(data)
    missing_data = missing_data.astype(float)
    if random_rate != 0:
        missing_data = np.copy(data)
        missing_indices = np.random.choice(data.shape[0], int(data.shape[0] * random_rate), replace=False)
        missing_data[missing_indices] = np.nan
        return missing_data
    if drop_array is not None:
        missing_data[drop_array] = np.nan
        return missing_data
    return missing_data

def create_missing_data(data, random_rate=0, drop_array=None):
    data_new = []
    print(f"debug: data_shape: {data.shape}")
    for idx, traj in enumerate(data):
        # print("debug: traj_shape: ", traj.shape)
        data_new.append(torch.Tensor(missing_data_single(traj.squeeze(), random_rate=random_rate, drop_array=drop_array)).unsqueeze(1).to(parameters.device))
        # print("debug: after processing: ", data_new[-1].shape)
    data_new = np.array(data_new)
    data_new = torch.tensor(data_new).to(parameters.device)
    print(f"debug: data_new_shape: {data_new.shape}")
        
    return data_new
    

def remove_nan(data):
    """
    Remove NaN values from the input data.
    
    Args:
        data (np.ndarray): Input data.
    
    Returns:
        np.ndarray: Data with NaN values removed.
    """
    return data[~np.isnan(data)]

def modify_data():
    """
    Modifies parameters file dataset to instead run with missing data.
    """
    print("Logs: data_dropout: Modifying data to include missing data.")
    trajs = parameters.dataset['trajs']
    times = parameters.dataset['times']
    
    new_trajs = create_missing_data(trajs, drop_array=parameters.dataset['missing_idx'])
    new_times = create_missing_data(times, drop_array=parameters.dataset['missing_idx'])

    if parameters.trainer == 'B-VAE':
        parameters.dataset['trajs'] = torch.Tensor(remove_nan(new_trajs)).to(parameters.device)
        parameters.dataset['times'] = torch.Tensor(remove_nan(new_times)).to(parameters.device)
    elif parameters.trainer == 'RNN' or parameters.trainer == 'LSTM':
        parameters.dataset['trajs'] = torch.Tensor(new_trajs).to(parameters.device)
        parameters.dataset['times'] = torch.Tensor(new_times).to(parameters.device)
    else:
        print("modification of data not available for this trainer: ", parameters.trainer)
        print("no modifications made.")
        

if __name__ == '__main__':
    data = [0,1,2,3,4,5,6,7,8,9,10]
    missing_data_index = [1, 5, 9]
    new_data = create_missing_data(data, drop_array=missing_data_index)
    expected_data = [0, np.nan, 2, 3, 4, np.nan, 6, 7, 8, np.nan, 10]
    expected_data = np.array(expected_data)
    expected_data = expected_data.astype(float)
    if not np.array_equal(new_data, expected_data):
        print("manual inspection: ")
        print("Expected: ", expected_data)
        print("Got: ", new_data)
    else:
    
        print("Test passed!")