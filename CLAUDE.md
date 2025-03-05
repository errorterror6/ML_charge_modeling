# ML Charge Modeling Project Guidelines

## Environment Setup
- `pipenv install` - Install dependencies
- `pipenv shell` - Activate virtual environment

## Commands
- `python libs/shjnn/train.py` - Train the model
- `python libs/shjnn/test.py` - Run model inference
- `python parameter_tuning/main.py` - Run parameter tuning

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

## Project Structure
- `/libs` - Core implementation modules
- `/nbks` - Jupyter notebooks for analysis
- `/parameter_tuning` - Parameter optimization code
- `/data` - Data storage (excluded from version control)