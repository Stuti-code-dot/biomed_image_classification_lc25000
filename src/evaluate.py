import argparse, yaml, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from dataset import ImageFolderWithPaths
from model import build_model
from utils import get_device

@torch.no_grad()
def run_eval(model, ds, batch_size: int = 32, num_workers: int = 4, device: str = 'cpu'):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_probs, all_preds, all_targs = [], [], []
    model.eval()
    for x,y,_ in dl:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)
        all_probs.append(probs); all_preds.append(preds); all_targs.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targs, axis=0)
    ce = float(log_loss(targs, probs, labels=list(range(probs.shape[1]))))
    acc = float(accuracy_score(targs, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(targs, preds, average='macro', zero_division=0)
    cm = confusion_matrix(targs, preds).tolist()
    return {'ce': ce, 'accuracy': acc, 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'cm': cm}

def main(cfg_path):
    with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
    device = get_device()
    img_size = int(cfg['data']['img_size']); num_classes = int(cfg['data']['num_classes']); class_names = cfg['data']['class_names']
    model_name = cfg['train']['model_name']
    outdir = Path(cfg['output']['dir']); ckpt = outdir/'models'/f"{model_name}_best.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}. Train first."

    ds_val = ImageFolderWithPaths(cfg['data']['val_dir'], img_size, augment=False, class_names=class_names)
    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    metrics = run_eval(model, ds_val, batch_size=int(cfg['train']['batch_size']), num_workers=int(cfg['train']['num_workers']), device=device)
    print('Validation metrics:', metrics)
    (outdir/'metrics').mkdir(parents=True, exist_ok=True)
    with open(outdir/'metrics'/'val_metrics.json','w') as f:
        json.dump(metrics, f, indent=2)

    test_dir = Path(cfg['data']['test_dir'])
    if test_dir.exists():
        ds_test = ImageFolderWithPaths(str(test_dir), img_size, augment=False, class_names=class_names)
        test_metrics = run_eval(model, ds_test, batch_size=int(cfg['train']['batch_size']), num_workers=int(cfg['train']['num_workers']), device=device)
        print('Test metrics:', test_metrics)
        with open(outdir/'metrics'/'test_metrics.json','w') as f:
            json.dump(test_metrics, f, indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    args = ap.parse_args()
    main(args.config)
