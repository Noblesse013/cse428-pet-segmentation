import json

with open('multi-task-pet-segmentation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Markdown for Bonus Tasks
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## Bonus Tasks\n",
        "\n",
        "### Bonus Task 2: Data Augmentation Comparison\n",
        "We will train the base U-Net without data augmentation and compare it to the previously trained model (which used augmentation).\n",
        "\n",
        "### Bonus Task 4: Hyperparameter Tuning\n",
        "We will test different values for `weight_decay` to see how it affects overfitting."
    ]
}
nb['cells'].append(md_cell)

# Cell 2: Code for Bonus Task 2
code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import train_unet\n",
        "import time\n",
        "import pandas as pd\n",
        "\n",
        "print('Training Base U-Net WITHOUT Data Augmentation...')\n",
        "start = time.time()\n",
        "\n",
        "no_aug_history = train_unet.main(\n",
        "    num_epochs=config.NUM_EPOCHS,\n",
        "    batch_size=config.BATCH_SIZE,\n",
        "    checkpoint_dir=config.CHECKPOINT_DIR / 'no_aug',\n",
        "    results_dir=config.RESULTS_DIR / 'no_aug',\n",
        "    use_augmentation=False\n",
        ")\n",
        "\n",
        "print(f'\\nFinished in {(time.time() - start) / 60:.1f} min')\n"
    ]
}
nb['cells'].append(code_cell_2)

# Cell 3: Code for comparing Bonus Task 2
code_cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "best_aug_iou = max(unet_history['val_iou'])\n",
        "best_no_aug_iou = max(no_aug_history['val_iou'])\n",
        "print(f'Best Val IoU WITH Augmentation:    {best_aug_iou:.4f}')\n",
        "print(f'Best Val IoU WITHOUT Augmentation: {best_no_aug_iou:.4f}')\n",
        "print(f'Improvement from Augmentation:     {best_aug_iou - best_no_aug_iou:.4f}')\n"
    ]
}
nb['cells'].append(code_cell_3)

# Cell 4: Code for Bonus Task 4
code_cell_4 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print('Testing different weight_decay values for Hyperparameter Tuning...')\n",
        "wd_values = [1e-4, 1e-3]\n",
        "wd_histories = {}\n",
        "\n",
        "for wd in wd_values:\n",
        "    print(f'\\n--- Training with weight_decay = {wd} ---')\n",
        "    history = train_unet.main(\n",
        "        num_epochs=config.NUM_EPOCHS,\n",
        "        batch_size=config.BATCH_SIZE,\n",
        "        checkpoint_dir=config.CHECKPOINT_DIR / f'wd_{wd}',\n",
        "        results_dir=config.RESULTS_DIR / f'wd_{wd}',\n",
        "        weight_decay=wd\n",
        "    )\n",
        "    wd_histories[wd] = history\n"
    ]
}
nb['cells'].append(code_cell_4)

# Cell 5: Code for displaying Bonus Task 4 results
code_cell_5 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "rows = []\n",
        "rows.append({'Weight Decay': '1e-5 (Baseline)', 'Best Val IoU': best_aug_iou})\n",
        "for wd, hist in wd_histories.items():\n",
        "    rows.append({'Weight Decay': str(wd), 'Best Val IoU': max(hist['val_iou'])})\n",
        "\n",
        "wd_comparison = pd.DataFrame(rows)\n",
        "display(wd_comparison)\n"
    ]
}
nb['cells'].append(code_cell_5)

with open('multi-task-pet-segmentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
