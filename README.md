# ML_charge_modeling

python version:

3.12.2 (ray)

python3 -m ./.venv
pip install -r requirements.txt  (requirements-linux.txt for linux)

start venv using source ./.venv/bin/activate for linux
./.venv/Scripts/activate for windows

bulk of code is in /parameter_tuning/ folder.

## Running the parameter tuning

To run the parameter tuning:

1. Navigate to the parameter_tuning directory:
   ```
   cd parameter_tuning
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. Follow the prompts:
   - Enter a name for the run (e.g., "test_run")
   - Enter a description (e.g., "Testing B-VAE performance")
   - Choose whether to use dropout by typing 'y' or 'n'
   - Select a model type from:
     - B-VAE (Beta-Variational Autoencoder)
     - RNN (Recurrent Neural Network)
     - LSTM (Long Short-Term Memory)
