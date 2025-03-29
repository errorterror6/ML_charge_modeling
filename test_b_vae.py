import sys
sys.path.append('/mnt/c/vscode/thesis/ML_charge_modeling/libs')
sys.path.append('/mnt/c/vscode/thesis/ML_charge_modeling')

import torch
from torchdiffeq import odeint

# Create a simplified version of the error-causing code
def test_batch_processing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock trajectories and time points
    batch_size = 5
    seq_len = 10
    feature_dim = 1
    latent_dim = 2
    
    # Create random batch tensors
    traj_tensor = torch.randn(batch_size, seq_len, feature_dim).to(device)
    time_tensor = torch.linspace(0, 1, seq_len).unsqueeze(-1).repeat(batch_size, 1, 1).to(device)
    
    # Mock prediction tensors (simulating output from infer_step)
    pred_x = torch.randn(batch_size, seq_len, feature_dim).to(device)
    
    # Print shapes for debugging
    print(f"traj_tensor shape: {traj_tensor.shape}")
    print(f"time_tensor shape: {time_tensor.shape}")
    print(f"pred_x shape: {pred_x.shape}")
    
    # Process individually as in the original code
    loss_list = []
    try:
        batch_size_actual = batch_size
        
        # Compute individual losses for each sample in batch
        for i in range(batch_size_actual):
            # Extract individual tensors
            pred_x_i = pred_x[i:i+1]
            traj_tensor_i = traj_tensor[i:i+1]
            
            # Print shapes
            print(f"pred_x_i shape: {pred_x_i.shape}")
            print(f"traj_tensor_i shape: {traj_tensor_i.shape}")
            
            # Compute loss (ensure on same device)
            loss = torch.nn.MSELoss()(pred_x_i.to(device), traj_tensor_i.to(device))
            loss_list.append(loss.item())
            
        print(f"Individual losses: {loss_list}")
        print(f"Mean loss: {sum(loss_list)/len(loss_list)}")
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_processing()