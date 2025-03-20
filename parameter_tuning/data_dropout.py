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

def verify_missing_data():
    count = 0;
    for entry in parameters.dataset['trajs']:
        for step in entry:
            if torch.isnan(step).any():
                count += 1

    for entry in parameters.dataset['times']:
        for step in entry:
            if torch.isnan(step).any():
                count += 1
    print("Logs: data_dropout: Number of missing data entries: ", count)
    

def remove_nan(data):
    """
    Truncate NaN values from the dataset while preserving memory structure.
    
    Args:
        data (np.ndarray or torch.Tensor): Input data, could be 2D or 3D tensor.
    
    Returns:
        np.ndarray or torch.Tensor: Data with NaN values removed.
    """
    if isinstance(data, torch.Tensor):
        # Check tensor dimensionality
        if data.dim() == 2:
            # For 2D tensor [batch, features]
            mask = ~torch.isnan(data).any(dim=1)
            return data[mask]
        elif data.dim() == 3:
            # For 3D tensor [batch, seq_len, features]
            # We need to create masks for each sequence item
            mask = ~torch.isnan(data).any(dim=2)  # Shape: [batch, seq_len]
            
            # For each batch item, keep only valid timesteps
            filtered_data_list = []
            
            for i in range(data.shape[0]):  # For each trajectory
                item_mask = mask[i]  # Get mask for this trajectory
                if item_mask.any():  # If there are any valid timesteps
                    filtered_data_list.append(data[i, item_mask])
            
            if filtered_data_list:
                # Stack the valid data back together
                return torch.stack(filtered_data_list)
            else:
                # If no valid data, return empty tensor with correct feature dimension
                return torch.zeros((0, data.shape[2]), device=data.device)
        else:
            raise ValueError(f"Unsupported tensor dimension: {data.dim()}")
    else:
        # For numpy arrays
        if data.ndim == 2:
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]
        elif data.ndim == 3:
            mask = ~np.isnan(data).any(axis=2)
            
            filtered_data_list = []
            for i in range(data.shape[0]):
                item_mask = mask[i]
                if item_mask.any():
                    filtered_data_list.append(data[i, item_mask])
            
            if filtered_data_list:
                return np.stack(filtered_data_list)
            else:
                return np.zeros((0, data.shape[2]))
        else:
            raise ValueError(f"Unsupported array dimension: {data.ndim}")

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
        # Use our improved remove_nan function which handles correspondence between datasets
        print("Removing NaN values from trajectories...")
        filtered_trajs = remove_nan(new_trajs)
        
        # Apply the same masking logic to times
        print("Removing corresponding time points...")
        # We need to identify NaN values in trajs to filter both trajs and times
        if isinstance(new_trajs, torch.Tensor):
            mask = ~torch.isnan(new_trajs).any(dim=2)  # Shape: [batch, seq_len]
            
            filtered_times_list = []
            for i in range(new_trajs.shape[0]):
                traj_mask = mask[i]
                if traj_mask.any():
                    filtered_times_list.append(new_times[i, traj_mask])
            
            if filtered_times_list:
                filtered_times = torch.stack(filtered_times_list)
            else:
                print("Warning: All trajectory points contain NaN values")
                filtered_times = new_times  # Fallback
        else:
            mask = ~np.isnan(new_trajs).any(axis=2)
            
            filtered_times_list = []
            for i in range(new_trajs.shape[0]):
                traj_mask = mask[i]
                if traj_mask.any():
                    filtered_times_list.append(new_times[i, traj_mask])
            
            if filtered_times_list:
                filtered_times = np.stack(filtered_times_list)
            else:
                print("Warning: All trajectory points contain NaN values")
                filtered_times = new_times  # Fallback
                
        # Update the dataset
        parameters.dataset['trajs'] = torch.Tensor(filtered_trajs).to(parameters.device)
        parameters.dataset['times'] = torch.Tensor(filtered_times).to(parameters.device)
        print(f"After NaN removal - trajs shape: {parameters.dataset['trajs'].shape}, times shape: {parameters.dataset['times'].shape}")
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