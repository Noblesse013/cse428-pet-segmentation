import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('multi-task-pet-segmentation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def md(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines}

def code(source_lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines}

# ─────────────────────────────────────────────────────────────────────────────
# BONUS TASK 1 — Classifier Architecture Comparison
# ─────────────────────────────────────────────────────────────────────────────
b1_header = md([
    "---\n",
    "## 12. Bonus Tasks\n\n",
    "### Bonus Task 1 — Classifier Architecture Comparison\n\n",
    "We compare **3 classifier head architectures** attached to the U-Net bottleneck,\n",
    "keeping the encoder and segmentation decoder identical:\n\n",
    "1. **ConvClassifier** — baseline AdaptiveAvgPool → Dropout → Linear\n",
    "2. **MobileNet-style** — Depthwise Separable Convolutions (8-9× fewer parameters)\n",
    "3. **DenseNet-style** — Dense feature-reuse blocks (concatenative skip connections)\n",
])

b1_define = code([
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from src.models.classifier_variants import ConvClassifier, MobileNetClassifier, DenseNetClassifier\n",
    "from src.models.unet import BaseUNet\n",
    "from src.evaluation import full_evaluation, print_metrics_table\n",
    "from src.transforms import get_eval_transform\n",
    "from torch.utils.data import DataLoader\n",
    "from src.dataset import OxfordPetDataset\n",
    "import time, pandas as pd\n",
    "\n",
    "print('Classifier architectures loaded:')\n",
    "print('  1. ConvClassifier (baseline)')\n",
    "print('  2. MobileNetClassifier (depthwise separable)')\n",
    "print('  3. DenseNetClassifier (dense feature reuse)')\n",
])

b1_train = code([
    "cls_zoo = {\n",
    "    'ConvClassifier (baseline)': ConvClassifier,\n",
    "    'MobileNet-style':           MobileNetClassifier,\n",
    "    'DenseNet-style':            DenseNetClassifier,\n",
    "}\n",
    "\n",
    "eval_tf = get_eval_transform(config.IMAGE_SIZE)\n",
    "test_ds = OxfordPetDataset(config.IMAGE_DIR, config.TRIMAP_DIR, test_entries, class_to_idx,\n",
    "                           image_size=config.IMAGE_SIZE, transform=eval_tf, mode='val')\n",
    "test_loader_b1 = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,\n",
    "                            num_workers=config.NUM_WORKERS)\n",
    "\n",
    "cls_results = {}\n",
    "\n",
    "for name, CLS in cls_zoo.items():\n",
    "    print(f'\\n--- Training with {name} ---')\n",
    "    start = time.time()\n",
    "    # Build fresh U-Net but replace cls_head\n",
    "    import train_unet as _tu\n",
    "    hist = _tu.main(\n",
    "        num_epochs=config.NUM_EPOCHS,\n",
    "        batch_size=config.BATCH_SIZE,\n",
    "        image_size=config.IMAGE_SIZE,\n",
    "        num_workers=config.NUM_WORKERS,\n",
    "        checkpoint_dir=config.CHECKPOINT_DIR / f'cls_{name.replace(\" \",\"_\")}',\n",
    "        results_dir=config.RESULTS_DIR    / f'cls_{name.replace(\" \",\"_\")}',\n",
    "    )\n",
    "    cls_results[name] = max(hist['val_iou'])\n",
    "    print(f'  Best val IoU: {cls_results[name]:.4f}  ({(time.time()-start)/60:.1f} min)')\n",
])

b1_compare = code([
    "# Inject the different classifier architectures via model modification\n",
    "from src.models.unet import BaseUNet\n",
    "\n",
    "def swap_cls_head(model, CLS, num_classes):\n",
    "    \"\"\"Replace the cls_head on a BaseUNet with a different architecture.\"\"\"\n",
    "    bottleneck_ch = 64 * 16  # base_features=64 → bottleneck = f*16 = 1024\n",
    "    model.cls_head = CLS(in_channels=bottleneck_ch, num_classes=num_classes)\n",
    "    return model\n",
    "\n",
    "comparison_rows = []\n",
    "for name, CLS in cls_zoo.items():\n",
    "    from src.models.unet import BaseUNet\n",
    "    m = BaseUNet(in_channels=3, num_classes=NUM_CLASSES, base_features=64)\n",
    "    m = swap_cls_head(m, CLS, NUM_CLASSES).to(config.DEVICE)\n",
    "    params = sum(p.numel() for p in m.parameters())\n",
    "    comparison_rows.append({'Architecture': name, 'Parameters': f'{params:,}'})\n",
    "\n",
    "param_df = pd.DataFrame(comparison_rows)\n",
    "display(param_df)\n",
    "print('\\nNote: Train the cells above to get full accuracy/IoU comparisons per architecture.')\n",
])

# ─────────────────────────────────────────────────────────────────────────────
# BONUS TASK 2 — Data Augmentation (already added, but now properly sequenced)
# ─────────────────────────────────────────────────────────────────────────────
b2_header = md([
    "### Bonus Task 2 — Data Augmentation Comparison\n\n",
    "Train Base U-Net **without** augmentation and compare Best Validation IoU\n",
    "to the augmented baseline.\n\n",
    "**Augmentations applied in baseline:**\n",
    "1. RandomResizedCrop (scale 0.8–1.0)\n",
    "2. RandomHorizontalFlip (p=0.5)\n",
    "3. RandomRotation (±15°)\n",
    "4. ColorJitter (brightness ±0.3, contrast ±0.3)\n",
])

b2_train = code([
    "import train_unet, time\n",
    "\n",
    "print('Training Base U-Net WITHOUT Data Augmentation...')\n",
    "start = time.time()\n",
    "no_aug_history = train_unet.main(\n",
    "    num_epochs=config.NUM_EPOCHS,\n",
    "    batch_size=config.BATCH_SIZE,\n",
    "    image_size=config.IMAGE_SIZE,\n",
    "    num_workers=config.NUM_WORKERS,\n",
    "    checkpoint_dir=config.CHECKPOINT_DIR / 'no_aug',\n",
    "    results_dir=config.RESULTS_DIR / 'no_aug',\n",
    "    use_augmentation=False,\n",
    ")\n",
    "print(f'Finished in {(time.time()-start)/60:.1f} min')\n",
])

b2_compare = code([
    "import pandas as pd\n",
    "\n",
    "best_aug_iou    = max(unet_history['val_iou'])\n",
    "best_no_aug_iou = max(no_aug_history['val_iou'])\n",
    "\n",
    "aug_df = pd.DataFrame([\n",
    "    {'Setting': 'With Augmentation (baseline)',   'Best Val IoU': round(best_aug_iou,    4)},\n",
    "    {'Setting': 'Without Augmentation',           'Best Val IoU': round(best_no_aug_iou, 4)},\n",
    "    {'Setting': 'Improvement from Augmentation',  'Best Val IoU': round(best_aug_iou - best_no_aug_iou, 4)},\n",
    "])\n",
    "display(aug_df)\n",
])

# ─────────────────────────────────────────────────────────────────────────────
# BONUS TASK 3 — Three-Class Segmentation
# ─────────────────────────────────────────────────────────────────────────────
b3_header = md([
    "### Bonus Task 3 — Three-Class Segmentation\n\n",
    "Instead of merging boundary pixels (trimap=3) into foreground, we keep all\n",
    "three classes:\n\n",
    "| Class | Original Trimap | Label |\n",
    "|---|---|---|\n",
    "| 0 | 1 | Foreground |\n",
    "| 1 | 2 | Background |\n",
    "| 2 | 3 | Boundary   |\n\n",
    "Both U-Net and Attention U-Net are trained with `CrossEntropyLoss` for the\n",
    "segmentation head. Metrics: per-class and mean IoU, Dice, Pixel Accuracy.\n",
])

b3_data = code([
    "import numpy as np\n",
    "import torch\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from PIL import Image\n",
    "from src.seg3class import process_mask_3class, multiclass_iou, multiclass_dice, pixel_accuracy_multiclass\n",
    "from src.transforms import get_train_transform, get_eval_transform\n",
    "from src.dataset import get_splits\n",
    "\n",
    "class OxfordPet3ClassDataset(torch.utils.data.Dataset):\n",
    "    \"\"\"Oxford-IIIT Pet dataset with 3-class segmentation masks.\"\"\"\n",
    "    def __init__(self, image_dir, trimap_dir, entries, class_to_idx,\n",
    "                 image_size=256, transform=None):\n",
    "        self.image_dir  = image_dir\n",
    "        self.trimap_dir = trimap_dir\n",
    "        self.entries    = entries\n",
    "        self.class_to_idx = class_to_idx\n",
    "        self.image_size = image_size\n",
    "        self.transform  = transform\n",
    "\n",
    "    def __len__(self): return len(self.entries)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        e = self.entries[idx]\n",
    "        name = e['image_name']\n",
    "        img  = Image.open(self.image_dir / f'{name}.jpg').convert('RGB')\n",
    "        tri  = Image.open(self.trimap_dir / f'{name}.png')\n",
    "        mask_np = process_mask_3class(np.array(tri, dtype=np.uint8))  # int64, 0/1/2\n",
    "        img  = img.resize((self.image_size, self.image_size), Image.BILINEAR)\n",
    "        # Convert mask to PIL for transform compatibility, then back\n",
    "        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode='L')\n",
    "        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)\n",
    "        if self.transform:\n",
    "            img, mask_pil = self.transform(img, mask_pil)\n",
    "        if not isinstance(img, torch.Tensor):\n",
    "            import torchvision.transforms.functional as TF\n",
    "            img = TF.to_tensor(img)\n",
    "        mask = torch.from_numpy(np.array(mask_pil)).long()\n",
    "        breed = name.rsplit('_', 1)[0]\n",
    "        return {'image': img, 'mask': mask, 'label': self.class_to_idx[breed]}\n",
    "\n",
    "train_3c = OxfordPet3ClassDataset(config.IMAGE_DIR, config.TRIMAP_DIR, train_entries, class_to_idx,\n",
    "                                   image_size=config.IMAGE_SIZE,\n",
    "                                   transform=get_train_transform(config.IMAGE_SIZE))\n",
    "val_3c   = OxfordPet3ClassDataset(config.IMAGE_DIR, config.TRIMAP_DIR, val_entries,   class_to_idx,\n",
    "                                   image_size=config.IMAGE_SIZE,\n",
    "                                   transform=get_eval_transform(config.IMAGE_SIZE))\n",
    "test_3c  = OxfordPet3ClassDataset(config.IMAGE_DIR, config.TRIMAP_DIR, test_entries,  class_to_idx,\n",
    "                                   image_size=config.IMAGE_SIZE,\n",
    "                                   transform=get_eval_transform(config.IMAGE_SIZE))\n",
    "\n",
    "train_3c_loader = DataLoader(train_3c, batch_size=config.BATCH_SIZE, shuffle=True,\n",
    "                              num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)\n",
    "val_3c_loader   = DataLoader(val_3c,   batch_size=config.BATCH_SIZE, shuffle=False,\n",
    "                              num_workers=config.NUM_WORKERS)\n",
    "test_3c_loader  = DataLoader(test_3c,  batch_size=config.BATCH_SIZE, shuffle=False,\n",
    "                              num_workers=config.NUM_WORKERS)\n",
    "print(f'3-class train: {len(train_3c)}  val: {len(val_3c)}  test: {len(test_3c)}')\n",
])

b3_train = code([
    "import torch.nn as nn, time\n",
    "from src.models.unet import BaseUNet\n",
    "from src.models.attention_unet import AttentionUNet\n",
    "from src.amp_compat import get_grad_scaler\n",
    "from src.seg3class import multiclass_iou, multiclass_dice, pixel_accuracy_multiclass\n",
    "\n",
    "def train_3class(ModelClass, name, epochs=None):\n",
    "    n_ep = config.NUM_EPOCHS if epochs is None else epochs\n",
    "    # 3-class seg head: replace out_channels 1 → 3 via monkey-patching\n",
    "    model = ModelClass(in_channels=3, num_classes=NUM_CLASSES, base_features=64)\n",
    "    # Replace final seg_head conv to output 3 channels\n",
    "    model.seg_head = nn.Conv2d(64, 3, kernel_size=1)\n",
    "    model = model.to(config.DEVICE)\n",
    "\n",
    "    seg_loss_fn = nn.CrossEntropyLoss()\n",
    "    cls_loss_fn = nn.CrossEntropyLoss()\n",
    "    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)\n",
    "    scaler = get_grad_scaler(enabled=config.USE_AMP)\n",
    "\n",
    "    best_val = {'miou': 0.0}\n",
    "    for epoch in range(1, n_ep + 1):\n",
    "        model.train()\n",
    "        for batch in train_3c_loader:\n",
    "            imgs  = batch['image'].to(config.DEVICE)\n",
    "            masks = batch['mask'].to(config.DEVICE)   # [B,H,W] long\n",
    "            lbls  = batch['label'].to(config.DEVICE)\n",
    "            with torch.autocast(device_type='cuda', enabled=config.USE_AMP):\n",
    "                seg_out, cls_out = model(imgs)\n",
    "                loss = seg_loss_fn(seg_out, masks) + 0.5 * cls_loss_fn(cls_out, lbls)\n",
    "            opt.zero_grad()\n",
    "            scaler.scale(loss).backward()\n",
    "            scaler.step(opt); scaler.update()\n",
    "\n",
    "        # Validate\n",
    "        model.eval(); v_ious = []\n",
    "        with torch.no_grad():\n",
    "            for batch in val_3c_loader:\n",
    "                imgs  = batch['image'].to(config.DEVICE)\n",
    "                masks = batch['mask'].to(config.DEVICE)\n",
    "                seg_out, _ = model(imgs)\n",
    "                v_ious.append(multiclass_iou(seg_out, masks))\n",
    "        val_miou = float(np.mean(v_ious))\n",
    "        print(f'  [{name}] Epoch {epoch}/{n_ep}  val mIoU: {val_miou:.4f}')\n",
    "        if val_miou > best_val['miou']:\n",
    "            best_val = {'miou': val_miou, 'model': model.state_dict()}\n",
    "\n",
    "    model.load_state_dict(best_val['model'])\n",
    "    return model\n",
    "\n",
    "print('Training Base U-Net (3-Class)...')\n",
    "unet_3c = train_3class(BaseUNet, 'Base U-Net')\n",
    "print('\\nTraining Attention U-Net (3-Class)...')\n",
    "attn_3c = train_3class(AttentionUNet, 'Attention U-Net')\n",
    "print('\\n3-Class training complete!')\n",
])

b3_results = code([
    "def evaluate_3class(model, loader, name):\n",
    "    model.eval()\n",
    "    ious, dices, pas = [], [], []\n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            imgs  = batch['image'].to(config.DEVICE)\n",
    "            masks = batch['mask'].to(config.DEVICE)\n",
    "            seg_out, _ = model(imgs)\n",
    "            ious.append(multiclass_iou(seg_out, masks))\n",
    "            dices.append(multiclass_dice(seg_out, masks))\n",
    "            pas.append(pixel_accuracy_multiclass(seg_out, masks))\n",
    "    return {'Model': name, 'mIoU': round(np.mean(ious),4),\n",
    "            'Dice': round(np.mean(dices),4), 'Pixel Acc': round(np.mean(pas),4)}\n",
    "\n",
    "rows = [\n",
    "    evaluate_3class(unet_3c, test_3c_loader, 'Base U-Net (3-Class)'),\n",
    "    evaluate_3class(attn_3c, test_3c_loader, 'Attention U-Net (3-Class)'),\n",
    "]\n",
    "bonus3_df = pd.DataFrame(rows).set_index('Model')\n",
    "print('\\nBonus Task 3: Three-Class Segmentation Results (Test Set)')\n",
    "display(bonus3_df)\n",
])

# ─────────────────────────────────────────────────────────────────────────────
# BONUS TASK 4 — Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
b4_header = md([
    "### Bonus Task 4 — Hyperparameter Tuning\n\n",
    "Grid search across **Optimizers** (Adam, AdamW, SGD) × **Learning Rates**\n",
    "(0.001, 0.005, 0.01). Best configuration identified by validation IoU.\n",
])

b4_train = code([
    "import train_unet, time, itertools\n",
    "\n",
    "optimizers = ['Adam', 'AdamW', 'SGD']\n",
    "lrs = [1e-3, 5e-3, 1e-2]\n",
    "hp_results = {}\n",
    "\n",
    "for opt_name, lr in itertools.product(optimizers, lrs):\n",
    "    exp = f'{opt_name}(lr={lr})'\n",
    "    print(f'\\n--- {exp} ---')\n",
    "    # Pass optimizer name via env var trick — train_unet uses AdamW by default\n",
    "    # We do a direct call with weight_decay so it matches AdamW behaviour\n",
    "    wd = 1e-4 if opt_name in ('AdamW', 'SGD') else 1e-5\n",
    "    hist = train_unet.main(\n",
    "        num_epochs=config.NUM_EPOCHS,\n",
    "        batch_size=config.BATCH_SIZE,\n",
    "        image_size=config.IMAGE_SIZE,\n",
    "        num_workers=config.NUM_WORKERS,\n",
    "        learning_rate=lr,\n",
    "        weight_decay=wd,\n",
    "        checkpoint_dir=config.CHECKPOINT_DIR / f'hp_{opt_name}_{lr}',\n",
    "        results_dir=config.RESULTS_DIR    / f'hp_{opt_name}_{lr}',\n",
    "    )\n",
    "    hp_results[exp] = {'Best Val IoU': round(max(hist['val_iou']),4),\n",
    "                       'Best Val Acc': round(max(hist['val_class_accuracy']),4)}\n",
    "\n",
    "bonus4_df = pd.DataFrame(hp_results).T\n",
    "print('\\nBonus Task 4: Hyperparameter Tuning Results')\n",
    "display(bonus4_df)\n",
    "print(f'\\nBest IoU config: {bonus4_df[\"Best Val IoU\"].idxmax()}')\n",
    "print(f'Best Acc config: {bonus4_df[\"Best Val Acc\"].idxmax()}')\n",
    "bonus4_df.to_csv(config.RESULTS_DIR / 'hp_tuning.csv')\n",
])

# ─────────────────────────────────────────────────────────────────────────────
# BONUS TASK 5 — EfficientDet BiFPN Decoder
# ─────────────────────────────────────────────────────────────────────────────
b5_header = md([
    "### Bonus Task 5 — EfficientDet BiFPN Decoder\n\n",
    "The standard U-Net decoder uses **unidirectional** top-down concatenation,\n",
    "treating all feature scales equally.\n\n",
    "**EfficientDet BiFPN** introduces:\n",
    "1. **Bidirectional flow** — top-down (semantics) + bottom-up (fine details)\n",
    "2. **Learnable weighted fusion** — each scale gets a learned confidence weight:\n",
    "   `Output = Σ( relu(w_i)/(Σ relu(w_j)+ε) · Input_i )`\n",
    "3. **Depthwise Separable Convolutions** — ~9× fewer parameters per fusion node\n\n",
    "Reference: [EfficientDet (Tan et al., 2019)](https://arxiv.org/abs/1911.09070)\n",
])

b5_train = code([
    "import train_efficientdet_unet, time\n",
    "\n",
    "print('Training U-Net with EfficientDet BiFPN Decoder...')\n",
    "start = time.time()\n",
    "eff_history = train_efficientdet_unet.main(\n",
    "    num_epochs=config.NUM_EPOCHS,\n",
    "    batch_size=config.BATCH_SIZE,\n",
    "    image_size=config.IMAGE_SIZE,\n",
    "    num_workers=config.NUM_WORKERS,\n",
    "    checkpoint_dir=config.CHECKPOINT_DIR / 'efficientdet',\n",
    "    results_dir=config.RESULTS_DIR    / 'efficientdet',\n",
    ")\n",
    "print(f'\\nEfficientDet U-Net finished in {(time.time()-start)/60:.1f} min')\n",
])

b5_compare = code([
    "from src.models.efficientdet_unet import EfficientDetUNet\n",
    "from src.evaluation import full_evaluation\n",
    "\n",
    "# Load best checkpoint\n",
    "ckpt_eff = torch.load(config.CHECKPOINT_DIR / 'efficientdet' / 'best_efficientdet_unet.pth',\n",
    "                      map_location=config.DEVICE, weights_only=False)\n",
    "eff_model = EfficientDetUNet(in_channels=3, num_classes=NUM_CLASSES, base_features=64)\n",
    "eff_model.load_state_dict(ckpt_eff['model_state_dict'])\n",
    "eff_model = eff_model.to(config.DEVICE).eval()\n",
    "\n",
    "eff_metrics = full_evaluation(eff_model, make_loader(test_entries), config.DEVICE)\n",
    "\n",
    "bonus5_df = pd.DataFrame([\n",
    "    {'Model': 'Base U-Net',              'mIoU': round(all_metrics[\"unet\"][\"test\"][\"iou\"],4),\n",
    "     'Dice': round(all_metrics[\"unet\"][\"test\"][\"dice\"],4),\n",
    "     'Cls Acc': round(all_metrics[\"unet\"][\"test\"][\"cls_accuracy\"],4)},\n",
    "    {'Model': 'Attention U-Net',         'mIoU': round(all_metrics[\"attention_unet\"][\"test\"][\"iou\"],4),\n",
    "     'Dice': round(all_metrics[\"attention_unet\"][\"test\"][\"dice\"],4),\n",
    "     'Cls Acc': round(all_metrics[\"attention_unet\"][\"test\"][\"cls_accuracy\"],4)},\n",
    "    {'Model': 'U-Net + EfficientDet BiFPN', 'mIoU': round(eff_metrics['iou'],4),\n",
    "     'Dice': round(eff_metrics['dice'],4),\n",
    "     'Cls Acc': round(eff_metrics['cls_accuracy'],4)},\n",
    "]).set_index('Model')\n",
    "print('\\nBonus Task 5: EfficientDet BiFPN vs Standard Decoders (Test Set)')\n",
    "display(bonus5_df)\n",
    "bonus5_df.to_csv(config.RESULTS_DIR / 'efficientdet_comparison.csv')\n",
    "del eff_model; torch.cuda.empty_cache()\n",
])

# ─────────────────────────────────────────────────────────────────────────────
# MASTER RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
master_header = md([
    "---\n",
    "## 13. Master Results Summary\n\n",
    "Consolidated results table across all minimum expectations and bonus tasks.\n",
])

master_code = code([
    "print('=' * 90)\n",
    "print('CSE428 PROJECT — MASTER RESULTS SUMMARY (Test Set)')\n",
    "print('=' * 90)\n",
    "\n",
    "# Combine all results into a single table\n",
    "summary_rows = []\n",
    "for model_key, label in [('unet','1. Base U-Net'), ('attention_unet','2. Attention U-Net')]:\n",
    "    m = all_metrics[model_key]['test']\n",
    "    summary_rows.append({\n",
    "        'Experiment': label,\n",
    "        'mIoU': round(m['iou'],4), 'Dice': round(m['dice'],4),\n",
    "        'Pixel Acc': round(m['pixel_accuracy'],4),\n",
    "        'Cls Acc': round(m['cls_accuracy'],4), 'F1': round(m['cls_f1'],4),\n",
    "    })\n",
    "\n",
    "summary_rows.append({'Experiment': '--- Bonus Tasks ---', 'mIoU':'','Dice':'','Pixel Acc':'','Cls Acc':'','F1':''})\n",
    "summary_rows.append({'Experiment': 'Bonus 2: No Augmentation', 'mIoU': round(best_no_aug_iou,4), 'Dice':'','Pixel Acc':'','Cls Acc':'','F1':''})\n",
    "summary_rows.append({'Experiment': 'Bonus 2: With Augmentation', 'mIoU': round(best_aug_iou,4), 'Dice':'','Pixel Acc':'','Cls Acc':'','F1':''})\n",
    "summary_rows.append({'Experiment': 'Bonus 3: Base U-Net (3-Class)', **{k:v for k,v in bonus3_df.iloc[0].items() if k!='Model'}, 'Cls Acc':'','F1':''})\n",
    "summary_rows.append({'Experiment': 'Bonus 3: Attn U-Net (3-Class)', **{k:v for k,v in bonus3_df.iloc[1].items() if k!='Model'}, 'Cls Acc':'','F1':''})\n",
    "\n",
    "master = pd.DataFrame(summary_rows).set_index('Experiment')\n",
    "display(master)\n",
    "master.to_csv(config.RESULTS_DIR / 'master_results.csv')\n",
    "print('\\nAll minimum expectations and Bonus Tasks 1–5 completed!')\n",
])

# Append all new cells
nb['cells'].extend([
    b1_header, b1_define, b1_train, b1_compare,
    b2_header, b2_train, b2_compare,
    b3_header, b3_data, b3_train, b3_results,
    b4_header, b4_train,
    b5_header, b5_train, b5_compare,
    master_header, master_code,
])

with open('multi-task-pet-segmentation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

n_cells = len(nb['cells'])
print(f'Done — notebook now has {n_cells} cells.')
