import json
with open('multi-task-pet-segmentation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
code = []
for c in nb.get('cells', []):
    if c.get('cell_type') == 'code':
        exec_count = c.get('execution_count', '')
        source = ''.join(c.get('source', []))
        code.append(f"In [{exec_count}]:\n{source}")
with open('notebook_code.md', 'w', encoding='utf-8') as f:
    f.write('\n\n---\n\n'.join(code))
