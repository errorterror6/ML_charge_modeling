# SHJNN: Stochastic Hidden Jump Neural Network Library

## Overview
The SHJNN library implements a neural ordinary differential equation (ODE) based model for modeling dynamics in latent space, specifically designed for charge modeling applications. It provides a complete framework for training, inference, and analysis of time series data using latent ODE techniques.

## Module Structure

### `model.py` - Core Model Architecture
- **Functions**:
  - `init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, nbatch, lr, device=None)`: Initializes all model components and optimizer
- **Classes**:
  - `RecognitionRNN`: Encoder network that processes trajectories into latent space
    - Methods: `forward(observation, hidden_state)`, `initHidden()`
  - `LatentODEfunc`: Neural ODE dynamics function for latent space evolution
    - Methods: `forward(t, latent_state)`
  - `Decoder`: Network for transforming latent states to observation space
    - Methods: `forward(latent_state)`
  - `CDEFunc`: Neural CDE function (experimental)
    - Methods: `forward(z)`
  - `NeuralCDE`: Complete CDE model (experimental)
    - Methods: `forward(times, coeffs)`

### `data.py` - Data Processing
- **Classes**:
  - `CustomDataset`: PyTorch dataset for pairing trajectories with time points
    - Methods: `__init__(trajectories, time_points)`, `__len__()`, `__getitem__(idx)`
- **Functions**:
  - `prep_data()`: Loads, processes, normalizes data and converts to PyTorch tensors

### `train.py` - Training Functionality
- **Functions**:
  - `make_train_step(dynamics_func, recognition_network, decoder, optimizer, device, noise_std=0.3, beta=1.0)`: Factory function returning a complete training step function
  - `log_normal_pdf(x, mean, logvar)`: Computes log probability density of normal distribution
  - `normal_kl(mu1, lv1, mu2, lv2)`: Computes KL divergence between two normal distributions

### `infer.py` - Inference Functionality
- **Functions**:
  - `make_infer_step(dynamics_func, recognition_network, decoder, optimizer, device, input_mode='traj', sample=True)`: Factory function for inference with trained model
    - Supports two modes: `latent` input or `traj` (trajectory) input

### `orch.py` - High-level Orchestration
- **Functions**:
  - `train(dynamics_func, recognition_network, decoder, optimizer, trajectories, time_points, num_epochs, batch_size, device, beta=None, save_checkpoint=None)`: Main training loop
  - `save_state(path, dynamics_func, recognition_network, decoder, optimizer, loss, epochs)`: Saves model state
  - `load_state(path, dynamics_func, recognition_network, decoder, optimizer, loss, epochs, dev='gpu')`: Loads model state

### `analysis.py` - Analysis Tools
- **Functions**:
  - `umap_embedding(dimensions, n_neighbours=15, min_dist=0.8, n_components=2)`: Performs dimensionality reduction using UMAP
  - `dimension_reduction(dimensions, n_neighbors=15, min_dist=0.1, n_components=2)`: Wrapper for dimensionality reduction
  - `get_2d_embedding(dimensions, n_neighbors=15, min_dist=0.8)`: Gets 2D embeddings for visualization

### `test.py` - Test Scripts
- **Classes**:
  - `Lambda`: Simple ODE function for test purposes
  - `ODEFunc`: Neural network ODE function for test purposes
  - `RunningAverageMeter`: Utility for tracking running averages
- **Functions**:
  - `get_batch()`: Samples mini-batch for training
  - `makedirs(dirname)`: Creates directories
  - `visualize(true_y, pred_y, odefunc, itr)`: Visualization for ODE trajectories

## Usage Examples

### Training a Model
```python
import shjnn

# Initialize model components
dynamics_func, recognition_network, decoder, optimizer, device = shjnn.init_model(
    latent_dim=4, 
    nhidden=20, 
    rnn_nhidden=25, 
    obs_dim=2, 
    nbatch=32, 
    lr=0.001
)

# Prepare data
trajectories, time_points, scaler = shjnn.prep_data()

# Train the model
epochs, loss_history, mse_loss_history, kl_loss_history = shjnn.train(
    dynamics_func, 
    recognition_network, 
    decoder, 
    optimizer, 
    trajectories, 
    time_points, 
    num_epochs=100, 
    batch_size=32, 
    device=device, 
    beta=1.0
)

# Save model state
shjnn.save_state("model_checkpoint.pth", dynamics_func, recognition_network, decoder, optimizer, loss_history, epochs)
```

### Inference with Trained Model
```python
import shjnn
import torch

# Load model state
dynamics_func, recognition_network, decoder, optimizer, device = shjnn.init_model(
    latent_dim=4, 
    nhidden=20, 
    rnn_nhidden=25, 
    obs_dim=2, 
    nbatch=1, 
    lr=0.001
)
epochs = shjnn.load_state("model_checkpoint.pth", dynamics_func, recognition_network, decoder, optimizer, [], 0)

# Create inference function
infer_step = shjnn.make_infer_step(
    dynamics_func, 
    recognition_network, 
    decoder, 
    optimizer, 
    device, 
    input_mode='traj', 
    sample=False
)

# Run inference on a trajectory
trajectory = torch.tensor(...).to(device)  # Your input trajectory
time_points = torch.tensor(...).to(device)  # Corresponding time points

pred_x, pred_z = infer_step(trajectory, time_points)
```

## Architecture
The SHJNN model combines:
1. A variational encoder (RecognitionRNN) that processes trajectories in reverse order
2. A neural ODE (LatentODEfunc) that models dynamics in latent space
3. A decoder network that maps latent states back to observation space

The model is trained using variational inference with a loss function combining reconstruction error and KL divergence.

## Dependencies
- PyTorch
- torchdiffeq
- numpy
- sklearn
- umap
- hdbscan
- matplotlib (for visualization)