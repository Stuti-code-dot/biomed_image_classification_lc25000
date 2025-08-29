import argparse, yaml
from pathlib import Path
import cv2, numpy as np, torch
from model import build_model
from utils import get_device

def preprocess(img_path: str, img_size: int):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = (img/255.0 - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    img = np.transpose(img, (2,0,1)).astype(np.float32)
    return torch.tensor(img).unsqueeze(0)

@torch.no_grad()
def main(cfg_path, image_path):
    with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
    device = get_device()
    img_size = int(cfg['data']['img_size']); num_classes = int(cfg['data']['num_classes']); class_names = cfg['data']['class_names']
    model_name = cfg['train']['model_name']
    ckpt = Path(cfg['output']['dir'])/'models'/f"{model_name}_best.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}. Train first."

    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    x = preprocess(image_path, img_size).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    pred_idx = int(prob.argmax())
    print({'pred_class': class_names[pred_idx], 'prob': float(prob[pred_idx])})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--image', type=str, required=True)
    args = ap.parse_args()
    main(args.config, args.image)
