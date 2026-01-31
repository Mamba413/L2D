# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch
import numpy as np
import os
from metrics import get_roc_metrics, get_precision_recall_metrics

from torch.optim.lr_scheduler import CosineAnnealingLR
import time
try:
    from transformers import AdamW
except:
    from torch.optim import AdamW

def evaluate_model(model, data, device):
    model.to(device)
    model.eval()
    loss = 0
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, desc="Evaluating"):
            text = batch
            output = model(text)
            loss += output['loss'].item()
            epoch_crit_train_original.extend([output['crit'][1].item()])
            epoch_crit_train_sampled.extend([output['crit'][3].item()])
            
        print(f"Total time: {time.time() - start_time:.4f}s")
        avg_loss = loss / len(eval_loader)
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    
    # print(f"val_loss: {avg_loss:.6f}")
    print(f"val_ROC_AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"val_Real_mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, val_Samples_mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
    print("="*10)
    
    results_dict = {
        "name": "rewrite_gpt_ada",
        'info': {'n_samples': len(epoch_crit_train_original)},
        'predictions': {'real': epoch_crit_train_original, 
                        'samples': epoch_crit_train_sampled},
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    }
    return results_dict


def train_dist(model, data, device, ckpt_dir='./ckpt', args=None):
    train_loader = DataLoader(data, batch_size=1, shuffle=True)
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=0,
                                  last_epoch=-1)

    scaler = GradScaler()
    model.to(device)

    # Number of iterations for gradient accumulation
    accumulation_steps = args.a
    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(device)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        start_time = time.time()
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            text = batch
            scheduler.step()
            with autocast():
                outputs_1 = model(text)
                epoch_crit_train_original.extend([outputs_1['crit'][1].item()])
                epoch_crit_train_sampled.extend([outputs_1['crit'][3].item()])
                loss += (outputs_1['loss'].to(torch.float32)) / accumulation_steps
            
            del outputs_1

            if ((i + 1) % accumulation_steps) == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

                if i % 100 == 0:
                    torch.cuda.empty_cache()

                epoch_losses.append(loss.item())
                loss = torch.tensor(0.0).to(device)
            epoch_losses.append(loss.item())
            i += 1
        print(f"Total time: {time.time() - start_time:.4f}s") 
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"Real mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, Samples mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
        epoch_avg_loss = np.mean(epoch_losses)
        
        epoch_crit_train_original, epoch_crit_train_sampled = [],[] # reset crit
        print(f"\nAverage Loss for Epoch {epoch}: {epoch_avg_loss}")
    
    if args.save_trained:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print('Saving model...')
        model.save_pretrained(ckpt_dir)
        print(f"Saved finetuned model to {ckpt_dir}")
    
    return model

