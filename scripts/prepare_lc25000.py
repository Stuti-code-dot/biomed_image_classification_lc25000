#!/usr/bin/env python
"""
Reorganize LC25000 raw folder into ImageFolder-style train/val/test directories.
Expected raw structure (after unzipping):
  lung_colon_image_set/
    colon_image_sets/{colon_aca, colon_n}/
    lung_image_sets/{lung_aca, lung_scc, lung_n}/
"""
import argparse, shutil
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

CLASS_MAP = {
    "colon_image_sets/colon_aca": "colon_aca",
    "colon_image_sets/colon_n":   "colon_n",
    "lung_image_sets/lung_aca":   "lung_aca",
    "lung_image_sets/lung_scc":   "lung_scc",
    "lung_image_sets/lung_n":     "lung_n",
}

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def gather_images(raw_dir: Path):
    files, labels = [], []
    for rel, cls in CLASS_MAP.items():
        d = raw_dir / rel
        for p in d.glob("*"):
            if p.suffix.lower() in IMG_EXTS:
                files.append(p)
                labels.append(cls)
    return files, labels

def split_and_copy(files, labels, out_dir: Path, val_ratio: float, test_ratio: float, seed: int = 42):
    out_train = out_dir / "train"
    out_val   = out_dir / "val"
    out_test  = out_dir / "test"
    for o in [out_train, out_val, out_test]:
        for cls in sorted(set(labels)):
            (o/cls).mkdir(parents=True, exist_ok=True)

    # First split off test, then val from the remainder
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx_all = list(range(len(files)))
    train_idx, test_idx = next(sss1.split(idx_all, labels))
    files_train = [files[i] for i in train_idx]
    labels_train = [labels[i] for i in train_idx]
    files_test = [files[i] for i in test_idx]
    labels_test = [labels[i] for i in test_idx]

    # val from train remainder
    val_size = val_ratio / (1.0 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx_sub = list(range(len(files_train)))
    tr_idx, val_idx = next(sss2.split(idx_sub, labels_train))
    files_tr = [files_train[i] for i in tr_idx]
    labels_tr = [labels_train[i] for i in tr_idx]
    files_val = [files_train[i] for i in val_idx]
    labels_val = [labels_train[i] for i in val_idx]

    def copy_bulk(src_files, src_labels, dst_root: Path):
        import shutil
        for p, lab in zip(src_files, src_labels):
            dst = dst_root / lab / p.name
            shutil.copy2(p, dst)

    copy_bulk(files_tr, labels_tr, out_train)
    copy_bulk(files_val, labels_val, out_val)
    copy_bulk(files_test, labels_test, out_test)

def main(raw_dir: str, out_dir: str, val_ratio: float, test_ratio: float):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, labels = gather_images(raw)
    print(f"Found {len(files)} images.")
    split_and_copy(files, labels, out, val_ratio, test_ratio)
    print(f"Prepared dataset at {out} with train/val/test splits.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="Path to raw 'lung_colon_image_set' folder")
    ap.add_argument("--out_dir", type=str, default="data/lc25000", help="Output folder for processed dataset")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()
    main(args.raw_dir, args.out_dir, args.val_ratio, args.test_ratio)
