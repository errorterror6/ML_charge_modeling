# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ML Charge Modeling Project Guide

## Project Overview
This project provides tools for modeling and analyzing charge extraction dynamics in photovoltaic devices using machine learning techniques. It includes several model architectures:
- Beta-Variational Autoencoder (B-VAE)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Autoencoder variants (MLP-VAE, RNN-VAE, LSTM-VAE)

## Environment Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA drivers (optional, for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ML_charge_modeling

# Install dependencies using pipenv
pipenv install
pipenv shell

# Alternative: use pip with requirements file
pip install -r requirements.txt  # For most systems
pip install -r requirements-linux.txt  # For Linux-specific dependencies
```

## Running the Project

### Core Models (SHJNN)
From the project root directory:
```bash
# Train the model
python libs/shjnn/train.py

# Run inference
python libs/shjnn/test.py
```

### Parameter Tuning
IMPORTANT: Parameter tuning must be run from within the parameter_tuning directory.

```bash
# Navigate to parameter_tuning directory
cd parameter_tuning

# Run the parameter tuning script
python main.py  # or python3 main.py
```

When running the parameter tuning process, you'll be prompted to:
1. Enter a name for the run (e.g., "test_run")
2. Enter a description (e.g., "Testing B-VAE performance")
3. Choose whether to use dropout by typing 'y' or 'n'
4. Select a model type:
   - B-VAE (Beta-Variational Autoencoder)
   - RNN (Recurrent Neural Network)
   - LSTM (Long Short-Term Memory)
   - MLP-VAE (Multi-Layer Perceptron VAE)
   - RNN-VAE (Recurrent Neural Network VAE)
   - LSTM-VAE (Long Short-Term Memory VAE)

### Jupyter Notebooks
Exploratory data analysis and visualization notebooks are in the `/nbks` directory:
```bash
# Start Jupyter from project root
jupyter notebook nbks/
```

Key notebooks:
- `parameter_tuning_PV.ipynb` - Parameter tuning for PV data
- `parameter_tuning_charge_extraction.ipynb` - Parameter tuning for charge extraction
- `trce-lode-rebuild.ipynb` - Time-resolved charge extraction with latent ODEs

## Testing

### Running Tests
```bash
# Run all SHJNN tests
python libs/shjnn/tests/run_tests.py

# Run parameter tuning tests
python parameter_tuning/tests/run_tests.py

# Run a specific test file
python -m unittest libs/shjnn/tests/test_data.py

# Run a specific test method
python -m unittest libs/shjnn/tests/test_data.py::TestClassName.test_method
```

## Project Structure
- `/libs` - Core implementation modules
  - `/libs/shjnn` - SHJNN model implementation
  - `/libs/data.py` - Data loading and preprocessing utilities
- `/nbks` - Jupyter notebooks for analysis
- `/parameter_tuning` - Parameter optimization code
- `/data` - Data storage (excluded from version control)
- `/docs` - Documentation, research papers, and notes
- `/run` - Output directories for experimental runs

## GPU Acceleration
GPU acceleration is supported for PyTorch models:
```bash
# Set specific GPU (if multiple are available)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
# OR
export CUDA_VISIBLE_DEVICES=1  # Use second GPU

# Force CPU usage
export CUDA_VISIBLE_DEVICES=-1
```

## Troubleshooting
- If you encounter import errors when running parameter tuning, make sure you're running from the parameter_tuning directory
- For CUDA issues, verify installation with `torch.cuda.is_available()`
- Memory errors might require reducing batch size in `parameters.py`
- For data shape issues, add debug prints of tensor shapes before and after transformations
- When working with autoencoders, ensure data dimensions match model expectations (especially in compile_stacked_data)
- For autoencoder evaluation, note that eval_loss_fn only uses the first 2 dimensions of feature vectors

## Code Style Guidelines
- 4-space indentation
- Snake_case for functions and variables
- CamelCase for classes
- Imports: standard library first, then third-party, then local
- Type hints encouraged for function signatures
- Docstrings: multiline triple-quoted strings
- Section headers with triple quotes (''' SECTION NAME ''')
- Comprehensive error handling with appropriate logging
- Keep functions focused and under 50 lines when possible
- Log messages should be prefixed with "logs: [module]: [function]:" for consistency
- Expected tensor shapes should be documented in comments
- Data processing functions should validate input shapes and types