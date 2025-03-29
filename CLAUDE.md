# ML Charge Modeling Project Guidelines

## Environment Setup
- `pipenv install` - Install dependencies
- `pipenv shell` - Activate virtual environment

## Commands
- `python libs/shjnn/train.py` - Train the model
- `python libs/shjnn/test.py` - Run model inference
- `python parameter_tuning/main.py` - Run parameter tuning

## Running Parameter Tuning
To run the parameter tuning process:
1. Execute `python parameter_tuning/main.py`
2. At the prompts:
   - Enter a name for the run (e.g., "test_run")
   - Enter a description (e.g., "Testing B-VAE performance")
   - Choose whether to use dropout by typing 'y' or 'n'
   - Select a model type from:
     - B-VAE (Beta-Variational Autoencoder)
     - RNN (Recurrent Neural Network)
     - LSTM (Long Short-Term Memory)

## Testing
- `python libs/shjnn/tests/run_tests.py` - Run all SHJNN tests
- `python parameter_tuning/tests/run_tests.py` - Run parameter tuning tests
- `python -m unittest libs/shjnn/tests/test_data.py` - Run specific test file
- `python -m unittest libs/shjnn/tests/test_data.py::TestClassName.test_method` - Run single test

## Code Style
- 4-space indentation
- Snake_case for functions and variables
- CamelCase for classes
- Imports: standard library first, then third-party, then local
- Type hints encouraged for function signatures
- Docstrings: multiline triple-quoted strings
- Section headers with triple quotes (''' SECTION NAME ''')
- Use relative imports within the project
- Comprehensive error handling with appropriate logging
- Keep functions focused and under 50 lines when possible

## GPU Acceleration
- GPU acceleration is supported for PyTorch models
- Set environment variable: `export CUDA_VISIBLE_DEVICES=0` to use specific GPU

## Project Structure
- `/libs` - Core implementation modules
- `/nbks` - Jupyter notebooks for analysis
- `/parameter_tuning` - Parameter optimization code
- `/data` - Data storage (excluded from version control)