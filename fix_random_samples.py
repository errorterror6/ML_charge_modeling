import json
import os
import glob

# Function to update a single notebook
def update_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse {notebook_path} as JSON")
            return 0
    
    count = 0
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            # Process each line in the cell
            for i, line in enumerate(cell.get('source', [])):
                if 'randomsamples' in line and 'random_samples' not in line:
                    count += 1
                    modified_line = line.replace('randomsamples', 'random_samples')
                    cell['source'][i] = modified_line
                    print(f"Original: {line.strip()}")
                    print(f"Modified: {modified_line.strip()}")
    
    # Save the modified notebook
    if count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Updated {count} instances in {os.path.basename(notebook_path)}")
    else:
        print(f"No updates needed in {os.path.basename(notebook_path)}")
    
    return count

# Find all notebooks
notebooks = glob.glob('/mnt/c/vscode/thesis/ML_charge_modeling/nbks/**/*.ipynb', recursive=True)
total_updates = 0

# Process each notebook
for notebook_path in notebooks:
    print(f"\nProcessing {os.path.basename(notebook_path)}...")
    total_updates += update_notebook(notebook_path)

print(f"\nComplete\! Made {total_updates} updates across all notebooks.")
