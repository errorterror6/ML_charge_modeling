import json
import os
import glob

# Function to check a single notebook
def check_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse {notebook_path} as JSON")
            return 0
    
    lines_to_fix = []
    # Process each cell
    for cell_index, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            # Process each line in the cell
            for line_index, line in enumerate(cell.get('source', [])):
                if '_input' in line or '_sample' in line:
                    lines_to_fix.append((cell_index, line_index, line.strip()))
    
    if lines_to_fix:
        print(f"Found {len(lines_to_fix)} line(s) that still need fixing in {os.path.basename(notebook_path)}:")
        for cell_index, line_index, line in lines_to_fix:
            print(f"  Cell {cell_index}, Line {line_index}: {line}")
    else:
        print(f"No issues found in {os.path.basename(notebook_path)}")
    
    return len(lines_to_fix)

# Find all notebooks
notebooks = glob.glob('/mnt/c/vscode/thesis/ML_charge_modeling/nbks/**/*.ipynb', recursive=True)
total_issues = 0

# Process each notebook
for notebook_path in notebooks:
    print(f"\nChecking {os.path.basename(notebook_path)}...")
    total_issues += check_notebook(notebook_path)

print(f"\nComplete\! Found {total_issues} total issues across all notebooks.")
