"""
Data Processing Module for Stochastic Hidden Jump Neural Network (SHJNN)

This module provides functions and classes for:
1. Loading and preparing experimental data
2. Preprocessing and normalizing data
3. Creating PyTorch datasets for model training
"""

# Standard library imports
import pickle
import os

# Data processing
import numpy as np
from sklearn import preprocessing

# PyTorch
import torch
from torch.utils.data import Dataset


def prep_data():
    """
    Prepare dataset for training.
    
    This function:
    1. Loads raw data from the database
    2. Filters and processes the data
    3. Normalizes features
    4. Converts to PyTorch tensors
    
    Returns:
        tuple: (trajectories, time_points, data_scaler)
            - trajectories: List of tensor trajectories [sequence_length, feature_dim]
            - time_points: List of tensor time points [sequence_length, 1]
            - data_scaler: StandardScaler for denormalizing predictions
    """
    # Import raw dataset
    base_path, file_name = '../data', 'db'
    
    # Open dataset file
    with open(os.path.join(base_path, file_name), 'rb') as file:
        database = pickle.load(file)

    # Get list of unique device IDs
    device_ids = list(set([entry['device'] for entry in database]))

    # Prepare lists to store processed data
    processed_data = []
    time_points = []

    # Define features to extract
    feature_keys = [
        'temperature',  # Environmental temperature
        'intensity',    # Light intensity
        'proc_time',    # Processing time
        'voc',          # Open-circuit voltage
        'ff',           # Fill factor
        'rs',           # Series resistance
    ]
    
    # Define time variable (dependent variable for sorting)
    time_var = 'proc_time'

    # Process data for each device
    for device_id in device_ids:
        # Find all measurements for current device
        device_measurements = [entry for entry in database if entry['device'] == device_id]
        
        # Sort measurements by processing time
        sort_indices = np.argsort(np.array([entry[time_var] for entry in device_measurements]))
        device_measurements = [device_measurements[i] for i in sort_indices]
        
        # Extract feature values and time points
        device_features = np.array([[entry[key] for key in feature_keys] 
                                  for entry in device_measurements])
        
        device_times = np.array([[entry[time_var]] for entry in device_measurements])
        
        # Only use devices with exactly 8 time points
        if len(device_times) == 8:
            processed_data.append(device_features)
            time_points.append(device_times)

    # Normalize dataset features
    # Fit scaler to all data (concatenated across devices)
    data_scaler = preprocessing.StandardScaler().fit(np.concatenate(processed_data))
    
    # Transform each device's data with the scaler
    normalized_data = [data_scaler.transform(device_data) for device_data in processed_data]

    # Convert to PyTorch tensors
    trajectory_tensors = [torch.Tensor(data) for data in normalized_data]
    time_tensors = [torch.Tensor(times) for times in time_points]

    # Return trajectories, time points, and the scaler for later use
    return trajectory_tensors, time_tensors, data_scaler


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for trajectories and time points.
    
    This dataset pairs trajectory data with corresponding time points
    for use with PyTorch DataLoader.
    
    Attributes:
        x (list): List of trajectory tensors
        y (list): List of time point tensors
    """

    def __init__(self, trajectories, time_points):
        """
        Initialize the dataset with trajectories and time points.
        
        Args:
            trajectories (list): List of trajectory tensors
            time_points (list): List of time point tensors
        """
        self.x = trajectories
        self.y = time_points

    def __getitem__(self, index):
        """
        Get a single data item.
        
        Args:
            index (int): Index of the item to get
            
        Returns:
            tuple: (trajectory, time_points) at the given index
        """
        return (self.x[index], self.y[index])

    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items
        """
        return len(self.x)