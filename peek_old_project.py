import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('cse428-project.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
# Cells 50-72 are the bonus tasks (0-indexed: 49-71)
bonus_cells = cells[49:]

for i, c in enumerate(bonus_cells):
    ctype = c['cell_type']
    src = ''.join(c.get('source', []))
    print(f"=== OLD Cell {i+50} [{ctype}] ===")
    print(src)
    print()
