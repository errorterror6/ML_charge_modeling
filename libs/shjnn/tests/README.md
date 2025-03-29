# SHJNN Library Tests

This directory contains unit tests for the SHJNN (Stochastic Hidden Jump Neural Network) library.

## Test Structure

- `test_model.py`: Tests for model components (LatentODEfunc, RecognitionRNN, Decoder)
- `test_data.py`: Tests for data handling components (CustomDataset)
- `test_train.py`: Tests for training functions (log_normal_pdf, normal_kl, make_train_step)
- `test_infer.py`: Tests for inference functions (make_infer_step with different modes)
- `test_orch.py`: Tests for orchestration functions (save_state, load_state)
- `run_tests.py`: Script to run all tests

## Running Tests

To run all tests:

```bash
# From the shjnn directory
python -m tests.run_tests
```

To run a specific test module:

```bash
# From the shjnn directory
python -m unittest tests.test_model
```

To run a specific test case:

```bash
# From the shjnn directory
python -m unittest tests.test_model.TestModelComponents.test_decoder
```

## Test Coverage

These tests verify:
- Model components initialize correctly and have expected shapes and behaviors
- Data handling functions work correctly
- Training and inference functions produce outputs with correct shapes
- Model state can be saved and loaded correctly

## Dependencies

The tests require:
- PyTorch
- NumPy
- torchdiffeq (for ODE solving)
- scikit-learn (for preprocessing)