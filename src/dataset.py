from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

class ImageFolderWithPaths(Dataset):
    def __init__(self, root: str, img_size: int, augment: bool, class_names: List[str] = None):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        self.labels = []
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if class_names is None:
            self.class_names = classes
        else:
            self.class_names = class_names
        cls_to_idx = {c:i for i,c in enumerate(self.class_names)}
        for c in self.class_names:
            for p in (self.root/c).glob('*'):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append(p)
                    self.labels.append(cls_to_idx[c])
        self.tf = self.build_tf(img_size, augment)

    def build_tf(self, img_size: int, augment: bool):
        if A is None:
            def f(img):
                img = cv2.resize(img, (img_size, img_size))
                img = img[:,:,::-1].transpose(2,0,1) / 255.0
                return torch.tensor(img, dtype=torch.float32)
            return f
        else:
            aug = []
            if augment:
                aug += [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                    A.RandomResizedCrop(img_size, img_size, scale=(0.8,1.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3,5), p=0.1),
                ]
            else:
                aug += [A.Resize(img_size, img_size)]
            aug += [A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()]
            return A.Compose(aug)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if A is None:
            x = self.tf(img)
        else:
            x = self.tf(image=img)['image']
        y = self.labels[idx]
        return x, y, str(path)

def make_loaders(train_dir:str, val_dir:str, img_size:int, batch_size:int, num_workers:int, class_names:List[str]):
    ds_tr = ImageFolderWithPaths(train_dir, img_size, augment=True, class_names=class_names)
    ds_va = ImageFolderWithPaths(val_dir, img_size, augment=False, class_names=class_names)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds_tr, ds_va, dl_tr, dl_va
