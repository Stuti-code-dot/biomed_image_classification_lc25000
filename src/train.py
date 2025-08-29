import argparse, yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from dataset import make_loaders, ImageFolderWithPaths
from model import build_model
from utils import set_seed, ensure_dir, save_json, get_device

def train_one_epoch(model, dl, loss_fn, opt, scaler, device, amp):
    model.train()
    losses, preds, targs = [], [], []
    pbar = tqdm(dl, desc='train', leave=False)
    for x,y,_ in pbar:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if amp:
            with autocast():
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            out = model(x); loss = loss_fn(out, y)
            loss.backward(); opt.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist()); targs.extend(y.cpu().numpy().tolist())
        pbar.set_postfix(loss=np.mean(losses))
    acc = accuracy_score(targs, preds)
    return float(np.mean(losses)), float(acc)

@torch.no_grad()
def validate(model, dl, loss_fn, device):
    model.eval()
    losses, preds, targs = [], [], []
    for x,y,_ in dl:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy().tolist())
        targs.extend(y.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(targs, preds)
    return float(np.mean(losses)), float(acc)

def main(cfg_path):
    with open(cfg_path,'r') as f: cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed',42))
    device = get_device()

    outdir = Path(cfg['output']['dir']); ensure_dir(outdir); ensure_dir(outdir/'models'); ensure_dir(outdir/'logs')
    img_size = int(cfg['data']['img_size'])
    bs = int(cfg['train']['batch_size']); epochs = int(cfg['train']['epochs']); lr = float(cfg['train']['lr'])
    wd = float(cfg['train']['weight_decay']); ls = float(cfg['train']['label_smoothing']); dr = float(cfg['train']['dropout'])
    amp = bool(cfg['train']['amp']); nw = int(cfg['train']['num_workers'])
    model_name = cfg['train']['model_name']; scheduler = cfg['train']['scheduler']; k_folds = int(cfg['train']['k_folds'])
    class_names = cfg['data']['class_names']; num_classes = int(cfg['data']['num_classes'])

    if k_folds <= 0:
        ds_tr, ds_va, dl_tr, dl_va = make_loaders(cfg['data']['train_dir'], cfg['data']['val_dir'], img_size, bs, nw, class_names)
        model = build_model(model_name, num_classes, dropout=dr).to(device)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scaler = GradScaler(enabled=amp)
        if scheduler == 'cosine':
            sch = CosineAnnealingLR(opt, T_max=epochs)
        elif scheduler == 'onecycle':
            steps_per_epoch = len(dl_tr)
            sch = OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            sch = None

        best_val = 1e9; best_path = outdir/'models'/f"{model_name}_best.pt"
        log = []
        patience = int(cfg['train']['early_stopping_patience']); wait=0
        for ep in range(1, epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, dl_tr, loss_fn, opt, scaler, device, amp)
            va_loss, va_acc = validate(model, dl_va, loss_fn, device)
            if sch is not None and scheduler != 'onecycle': sch.step()
            log.append({'epoch': ep, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})
            print(f"Epoch {ep:03d} | train CE {tr_loss:.4f} acc {tr_acc:.3f} | val CE {va_loss:.4f} acc {va_acc:.3f}")
            if va_loss < best_val:
                best_val = va_loss; wait=0
                if cfg['output']['save_best_only']:
                    torch.save(model.state_dict(), best_path)
            else:
                wait += 1
                if wait >= patience:
                    print('Early stopping.')
                    break
        save_json({'history': log, 'best_val_ce': best_val}, outdir/'logs'/'train_log.json')
        if not cfg['output']['save_best_only']:
            torch.save(model.state_dict(), best_path)
        print('Best checkpoint:', best_path)

    else:
        ds_all = ImageFolderWithPaths(cfg['data']['train_dir'], img_size, augment=True, class_names=class_names)
        X = list(range(len(ds_all))); y = [ds_all.labels[i] for i in range(len(ds_all))]
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=cfg.get('seed',42))
        fold_logs = []
        for fi, (tr_idx, va_idx) in enumerate(kf.split(X,y)):
            print(f"Fold {fi+1}/{k_folds}")
            tr_subset = torch.utils.data.Subset(ds_all, tr_idx)
            va_subset = torch.utils.data.Subset(ImageFolderWithPaths(cfg['data']['train_dir'], img_size, augment=False, class_names=class_names), va_idx)
            dl_tr = torch.utils.data.DataLoader(tr_subset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
            dl_va = torch.utils.data.DataLoader(va_subset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

            model = build_model(model_name, num_classes, dropout=dr).to(device)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
            opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
            scaler = GradScaler(enabled=amp)
            if scheduler == 'cosine':
                sch = CosineAnnealingLR(opt, T_max=epochs)
            elif scheduler == 'onecycle':
                steps_per_epoch = len(dl_tr)
                sch = OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            else:
                sch = None

            best_val = 1e9; best_path = outdir/'models'/f"{model_name}_fold{fi}.pt"
            patience = int(cfg['train']['early_stopping_patience']); wait=0
            for ep in range(1, epochs+1):
                tr_loss, tr_acc = train_one_epoch(model, dl_tr, loss_fn, opt, scaler, device, amp)
                va_loss, va_acc = validate(model, dl_va, loss_fn, device)
                if sch is not None and scheduler != 'onecycle': sch.step()
                print(f"[Fold {fi}] Epoch {ep:03d} | train CE {tr_loss:.4f} acc {tr_acc:.3f} | val CE {va_loss:.4f} acc {va_acc:.3f}")
                if va_loss < best_val:
                    best_val = va_loss; wait=0; torch.save(model.state_dict(), best_path)
                else:
                    wait += 1
                    if wait >= patience: break
            fold_logs.append({'fold': fi, 'best_val_ce': float(best_val)})
        save_json({'cv': fold_logs}, outdir/'logs'/'cv_log.json')
        print('CV done. Logs saved.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    args = ap.parse_args()
    main(args.config)
