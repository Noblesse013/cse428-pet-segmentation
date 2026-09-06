"""
Verify notebook files on disk.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paths = [
    ROOT / "notebooks" / "main" / "multi-task-pet-segmentation.ipynb",
    ROOT / "multi-task-pet-segmentation.ipynb",
]

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"File: {p.relative_to(ROOT)}")
    print(f"  Total Cells: {len(nb['cells'])}")
    print(f"  Title      : {nb['cells'][0]['source'][0].strip()}")
    print(f"  Subtitle   : {nb['cells'][0]['source'][1].strip()}")
    print(f"  Has import train_unet: {any('import train_unet' in str(c['source']) for c in nb['cells'])}")
    print(f"  Has git clone:         {any('git clone' in str(c['source']) for c in nb['cells'])}\n")
