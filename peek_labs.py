import json

def extract_cells(filepath):
    with open(filepath, encoding='utf-8') as f:
        nb = json.load(f)
    cells = []
    for c in nb.get('cells', []):
        cells.append({
            'type': c['cell_type'],
            'source': ''.join(c.get('source', []))
        })
    return cells

# Extract the Lab notebook that likely contains ML/DL code
labs = [
    'CSE428_Lab_3.ipynb',  # likely CNN/DL
    'CSE428_Lab_8.ipynb',  # largest after lab 3
    'CSE428_Pet_MultiTask_Colab_Kaggle.ipynb',  # most relevant
]

for lab in labs:
    cells = extract_cells(f'notebooks/{lab}')
    with open(f'notebooks/{lab.replace(".ipynb","_cells.txt")}', 'w', encoding='utf-8') as f:
        for i, c in enumerate(cells):
            f.write(f"=== Cell {i+1} [{c['type']}] ===\n")
            f.write(c['source'][:500])
            f.write('\n\n')
