import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('multi-task-pet-segmentation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Map of current markdown text fragment → replacement
rewrites = {
    # Cell 1 - Title
    "Base U-Net and Attention U-Net, each with a shared encoder driving **two heads**:": (
        "Two models are trained in this notebook: a **Base U-Net** and an **Attention U-Net**.\n"
        "Each model is trained to do two things at the same time:\n\n"
        "1. **Segmentation** — the pet is separated from the background.\n"
        "2. **Classification** — the breed of the pet is identified (37 classes).\n\n"
        "The notebook is run on Kaggle. A free GPU is used."
    ),
    "This notebook runs the project's real modules — it clones the repo rather than\n"
    "duplicating the code, so results here match `python train_unet.py` locally.": "",
    "### Before you run anything — turn the GPU on": "### Step 0 — Turn the GPU on before running",
    "## 1. Check the GPU": "## 1. GPU Check\nThe GPU is checked here. If no GPU is shown, go to `Settings → Accelerator → GPU T4`.",
    "## 2. Get the code": "## 2. Download the Code\nThe project code is downloaded from GitHub into the Kaggle environment.",
    "Tried in order: code already present (including anything you attached on Kaggle\nvia `+ Add Input`) -> `git clone` from `REPO_URL`. Point `REPO_URL` at your own\nfork if you have one, or set it to `\"\"` if you are supplying the code yourself.": (
        "The code is cloned automatically from GitHub. No manual steps are needed."
    ),
    "## 3. Dependencies": "## 3. Install Dependencies\nAll required Python packages are checked. Only missing ones are installed.",
    "Colab and Kaggle already ship PyTorch, torchvision, numpy, pandas, matplotlib,\nPillow, scikit-learn and tqdm — so this is normally a no-op. It installs only\nwhat is genuinely missing, which keeps the cell fast and avoids a runtime\nrestart from reinstalling torch.": (
        "Kaggle already has PyTorch and most libraries installed. This cell checks what is missing and installs it."
    ),
    "## 4. Fetch the dataset": "## 4. Download the Dataset\nThe Oxford-IIIT Pet dataset (~800 MB) is downloaded here.",
    "The images are gitignored (~800 MB), so they are downloaded here.\n\n"
    "- **Kaggle with an attached Oxford-IIIT Pet input:** detected automatically, nothing is downloaded.\n"
    "- **Otherwise:** both official tarballs are pulled from the Oxford VGG mirrors and extracted.\n\n"
    "Takes roughly 2–4 minutes on a first run; re-running the cell is instant.": (
        "If the dataset is already attached as a Kaggle input, nothing is downloaded.\n"
        "Otherwise, the images and annotations are downloaded from the Oxford VGG servers.\n"
        "This takes 2–4 minutes on the first run."
    ),
    "## 5. Configure the run": "## 5. Set Training Settings\nTraining settings are configured here. These values can be changed if needed.",
    "`config` resolves paths per environment and every value below can be changed here.\n"
    "On a free T4, one epoch of Base U-Net at 256×256 takes roughly **1–2 minutes**, so\n"
    "30 epochs per model is roughly 45–60 min each — about 1.5–2 hours for both,\n"
    "which sits well inside the session limits on either platform.\n\n"
    "**Start with `NUM_EPOCHS = 5` to confirm the whole notebook runs end to end**, then\n"
    "raise it for the real run.": (
        "Each epoch takes about 1–2 minutes on a free T4 GPU.\n"
        "30 epochs per model ≈ 1 hour.\n\n"
        "> **Tip:** Set `NUM_EPOCHS = 5` first to confirm everything runs, then increase to 30."
    ),
    "## 6. Explore the dataset": "## 6. Dataset Exploration\nThe dataset is explored here. A 3×3 grid of images is shown with their ground-truth masks overlaid.",
    "Splits, class balance, and a 3×3 grid of images with their ground-truth masks\n"
    "overlaid. Boundary pixels (trimap value 3) are folded into the foreground, so the\n"
    "mask is a clean binary pet/background target.": (
        "The data is split into train, validation, and test sets.\n"
        "Boundary pixels (trimap value 3) are merged into the foreground.\n"
        "The mask becomes a simple binary image: pet pixels = 1, background = 0."
    ),
    "## 7. Train the Base U-Net": "## 7. Train the Base U-Net\nThe Base U-Net is trained here. Training and validation loss/metrics are saved per epoch.",
    "Shared encoder → bottleneck, splitting into a skip-connected decoder (segmentation)\n"
    "and a pooled FC head (classification). The combined objective is\n\n"
    "    total = BCE + Dice + 1.0 * CrossEntropy\n\n"
    "The best checkpoint by **validation IoU** is saved as `best_unet.pth`.": (
        "The U-Net encoder is shared between two tasks:\n"
        "- A **segmentation decoder** produces the binary mask.\n"
        "- A **classification head** predicts the breed.\n\n"
        "The combined loss is: `total = BCE + Dice + CrossEntropy`\n\n"
        "The model checkpoint with the best validation IoU is saved automatically.\n\n"
        "> **Convergence:** A learning rate scheduler is used — the learning rate is reduced "
        "when validation IoU stops improving. Early stopping is also active: training stops "
        "if no improvement is seen for 7 consecutive epochs."
    ),
    "## 8. Train the Attention U-Net": "## 8. Train the Attention U-Net\nThe Attention U-Net is trained here. It is the same as the Base U-Net but with attention gates on the skip connections.",
    "Same encoder/decoder, but every skip connection passes through an additive\n"
    "attention gate (Oktay et al., 2018) that uses the deeper decoder feature as a\n"
    "gating signal to suppress background before concatenation.": (
        "Attention gates are added to each skip connection.\n"
        "The gate learns to focus on the pet and suppress background noise before the features are merged."
    ),
    "## 9. Evaluate both models on train / val / test": "## 9. Evaluate Both Models\nBoth models are evaluated on the training, validation, and test sets.",
    "Note that the training split is re-evaluated **without augmentation** here, so the\nnumbers are comparable across splits.": (
        "Evaluation is run without augmentation on all splits so that the numbers are directly comparable."
    ),
    "### Side-by-side comparison on the test set": "### Test Set Comparison\nThe test set results for both models are shown side by side.",
    "## 10. Qualitative demo": "## 10. Visual Predictions\nRandom images are passed through the trained model. The original image, true mask, and predicted mask are shown.",
    "Original image | ground-truth mask overlay | predicted mask overlay, titled with the\n"
    "true breed, the predicted breed, and the per-image IoU. Change `SAMPLE_INDICES` to\n"
    "inspect any images you like.": (
        "For each sample, three things are shown: the original image, the true mask, and the predicted mask.\n"
        "The true breed and predicted breed are also displayed.\n\n"
        "Change `SAMPLE_INDICES` to test with any image index."
    ),
    "## 11. Collect the outputs": "## 11. Download the Results\nAll results and plots are packed into a zip file for download.",
    "- **Kaggle:** everything under `/kaggle/working` is already saved as notebook\n"
    "  output — use the *Output* tab after committing the notebook.\n"
    "- **Colab:** this zips `results/` and `checkpoints/` and triggers a browser download.\n"
    "  Checkpoints are ~120 MB each, so set `INCLUDE_CHECKPOINTS = False` if you only\n"
    "  want the plots and CSVs.": (
        "On Kaggle, all files in `/kaggle/working` are automatically saved as notebook output.\n"
        "On Colab, a zip file is downloaded directly to the browser."
    ),
}

changed = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown':
        continue
    src = ''.join(cell['source'])
    for old, new in rewrites.items():
        if old in src:
            src = src.replace(old, new)
            changed += 1
    nb['cells'][i]['source'] = [src]

print(f'Rewrote {changed} markdown sections.')

with open('multi-task-pet-segmentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook saved.')
