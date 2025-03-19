import numpy as np
import torch

def create_missing_data(data, random_rate=0, drop_array=None):
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