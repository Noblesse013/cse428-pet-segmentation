import json
import shutil

# Use the clean notebook as the base
src = 'notebooks/CSE428_Pet_MultiTask_Colab_Kaggle.ipynb'
dst = 'multi-task-pet-segmentation.ipynb'

shutil.copy(src, dst)

with open(dst, encoding='utf-8') as f:
    nb = json.load(f)

# --- Bonus Section Markdown ---
md_header = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 12. Bonus Tasks\n",
        "\n",
        "### Bonus Task 2 — Data Augmentation Comparison\n",
        "We train the Base U-Net **without** data augmentation and compare the best validation IoU against the augmented model trained in Section 7.\n",
        "This empirically shows how much augmentation helps generalisation.\n",
        "\n",
        "### Bonus Task 4 — Hyperparameter Tuning (Weight Decay)\n",
        "We sweep `weight_decay` values (`1e-4`, `1e-3`) for the Adam optimiser and compare the best validation IoU\n",
        "against the baseline (`1e-5`). Higher weight decay adds stronger L2 regularisation and can close the\n",
        "train–validation gap."
    ]
}

# --- Bonus Task 2: train without augmentation ---
code_aug = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import train_unet, time\n",
        "\n",
        "print('Training Base U-Net WITHOUT Data Augmentation...')\n",
        "start = time.time()\n",
        "\n",
        "no_aug_history = train_unet.main(\n",
        "    num_epochs=config.NUM_EPOCHS,\n",
        "    batch_size=config.BATCH_SIZE,\n",
        "    image_size=config.IMAGE_SIZE,\n",
        "    num_workers=config.NUM_WORKERS,\n",
        "    checkpoint_dir=config.CHECKPOINT_DIR / 'no_aug',\n",
        "    results_dir=config.RESULTS_DIR / 'no_aug',\n",
        "    use_augmentation=False,\n",
        ")\n",
        "print(f'\\nFinished in {(time.time()-start)/60:.1f} min')\n"
    ]
}

# --- Bonus Task 2: comparison table ---
code_aug_compare = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "\n",
        "best_aug_iou    = max(unet_history['val_iou'])\n",
        "best_no_aug_iou = max(no_aug_history['val_iou'])\n",
        "\n",
        "aug_df = pd.DataFrame([\n",
        "    {'Setting': 'With Augmentation (baseline)',   'Best Val IoU': round(best_aug_iou, 4)},\n",
        "    {'Setting': 'Without Augmentation',           'Best Val IoU': round(best_no_aug_iou, 4)},\n",
        "    {'Setting': 'Improvement from Augmentation',  'Best Val IoU': round(best_aug_iou - best_no_aug_iou, 4)},\n",
        "])\n",
        "display(aug_df)\n"
    ]
}

# --- Bonus Task 4 markdown sub-header ---
md_wd = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Bonus Task 4 — Weight Decay Sweep\n"]
}

# --- Bonus Task 4: sweep training ---
code_wd = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import train_unet, time\n",
        "\n",
        "wd_values   = [1e-4, 1e-3]\n",
        "wd_histories = {}\n",
        "\n",
        "for wd in wd_values:\n",
        "    print(f'\\n--- weight_decay = {wd} ---')\n",
        "    start = time.time()\n",
        "    wd_histories[wd] = train_unet.main(\n",
        "        num_epochs=config.NUM_EPOCHS,\n",
        "        batch_size=config.BATCH_SIZE,\n",
        "        image_size=config.IMAGE_SIZE,\n",
        "        num_workers=config.NUM_WORKERS,\n",
        "        checkpoint_dir=config.CHECKPOINT_DIR / f'wd_{wd}',\n",
        "        results_dir=config.RESULTS_DIR    / f'wd_{wd}',\n",
        "        weight_decay=wd,\n",
        "    )\n",
        "    print(f'Finished in {(time.time()-start)/60:.1f} min')\n"
    ]
}

# --- Bonus Task 4: results table ---
code_wd_compare = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "\n",
        "rows = [{'Weight Decay': '1e-5 (baseline)', 'Best Val IoU': round(max(unet_history['val_iou']), 4)}]\n",
        "for wd, hist in wd_histories.items():\n",
        "    rows.append({'Weight Decay': str(wd), 'Best Val IoU': round(max(hist['val_iou']), 4)})\n",
        "\n",
        "wd_df = pd.DataFrame(rows)\n",
        "display(wd_df)\n",
        "wd_df.to_csv(config.RESULTS_DIR / 'weight_decay_comparison.csv', index=False)\n"
    ]
}

# Append all new cells
nb['cells'].extend([
    md_header,
    code_aug,
    code_aug_compare,
    md_wd,
    code_wd,
    code_wd_compare,
])

with open(dst, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Done — {dst} now has {len(nb['cells'])} cells.")
