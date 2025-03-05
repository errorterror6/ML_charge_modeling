import json
import os
import glob
import re

# Function to check a single notebook
def check_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse {notebook_path} as JSON")
            return 0
    
    infer_calls = []
    # Process each cell
    for cell_index, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            # Process each line in the cell
            for line_index, line in enumerate(cell.get('source', [])):
                if "make_infer_step" in line:
                    # Check if the old parameter names are still used
                    if "_input" in line or "_sample" in line:
                        infer_calls.append((cell_index, line_index, line.strip()))
    
    if infer_calls:
        print(f"Found {len(infer_calls)} make_infer_step call(s) with old parameter names in {os.path.basename(notebook_path)}:")
        for cell_index, line_index, line in infer_calls:
            print(f"  Cell {cell_index}, Line {line_index}: {line}")
    else:
        print(f"All make_infer_step calls in {os.path.basename(notebook_path)} use the new parameter names.")
    
    return len(infer_calls)

# Find all notebooks
notebooks = glob.glob('/mnt/c/vscode/thesis/ML_charge_modeling/nbks/**/*.ipynb', recursive=True)
total_issues = 0

# Process each notebook
for notebook_path in notebooks:
    print(f"\nChecking {os.path.basename(notebook_path)}...")
    total_issues += check_notebook(notebook_path)

print(f"\nComplete\! Found {total_issues} total make_infer_step calls with old parameter names.")
