import pathlib

attn_src = pathlib.Path('train_attention_unet.py').read_text(encoding='utf-8')

# 1. Add new params to main() signature
old_sig = '    use_augmentation=True,\n    weight_decay=1e-5,\n):'
new_sig = '    use_augmentation=True,\n    weight_decay=1e-5,\n    early_stopping_patience=7,\n    lr_scheduler_patience=3,\n    lr_scheduler_factor=0.5,\n):'
attn_src = attn_src.replace(old_sig, new_sig, 1)

# 2. Add scheduler + no_improve_count before training loop
old_opt = 'criterion = MultiTaskLoss(classification_weight=CLASSIFICATION_LOSS_WEIGHT)\n    scaler = get_grad_scaler(enabled=USE_AMP)\n\n    # \u2500\u2500 Training loop'
new_opt = ('scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n'
           '        optimizer, mode=\'max\', factor=lr_scheduler_factor,\n'
           '        patience=lr_scheduler_patience, verbose=True,\n'
           '    )\n'
           '    criterion = MultiTaskLoss(classification_weight=CLASSIFICATION_LOSS_WEIGHT)\n'
           '    scaler = get_grad_scaler(enabled=USE_AMP)\n'
           '    no_improve_count = 0\n\n'
           '    # \u2500\u2500 Training loop')
attn_src = attn_src.replace(old_opt, new_opt, 1)

# 3. Reset counter when best model is found
old_best = ('if val_metrics["iou"] > best_val_iou:\n'
            '            best_val_iou = val_metrics["iou"]\n'
            '            torch.save({')
new_best = ('if val_metrics["iou"] > best_val_iou:\n'
            '            best_val_iou = val_metrics["iou"]\n'
            '            no_improve_count = 0\n'
            '            torch.save({')
attn_src = attn_src.replace(old_best, new_best, 1)

# 4. Add scheduler step + early stopping after best-model save print
old_end = ('print(f"  [best] Saved best model (val IoU = {best_val_iou:.4f})")\n\n'
           '    # \u2500\u2500 Save history CSV')
new_end = ('print(f"  [best] Saved best model (val IoU = {best_val_iou:.4f})")\n'
           '        else:\n'
           '            no_improve_count += 1\n\n'
           '        scheduler.step(val_metrics["iou"])\n'
           '        current_lr = optimizer.param_groups[0]["lr"]\n'
           '        print(f"  [lr] current learning rate = {current_lr:.2e}")\n\n'
           '        if early_stopping_patience and no_improve_count >= early_stopping_patience:\n'
           '            print(f"  [early stop] No improvement for {early_stopping_patience} epochs. "\n'
           '                  f"Training stopped at epoch {epoch}.")\n'
           '            break\n\n'
           '    # \u2500\u2500 Save history CSV')
attn_src = attn_src.replace(old_end, new_end, 1)

pathlib.Path('train_attention_unet.py').write_text(attn_src, encoding='utf-8')

checks = ['ReduceLROnPlateau', 'early_stopping_patience', 'no_improve_count']
for c in checks:
    print(f'{c}: {"FOUND" if c in attn_src else "MISSING"}')
