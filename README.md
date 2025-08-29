# 🧬 Biomedical Image Classification Pipeline — LC25000

End-to-end PyTorch pipeline for **histopathology image classification** using the **LC25000** dataset (lung + colon cancer).
Targets **industry-relevant** modeling practices: **augmentations, cross-validation, hyperparameter tuning**, and robust evaluation.

> **Target performance**: With EfficientNet/ResNet + strong augmentations and CV, a **Cross-Entropy (CE) loss ~0.42** on validation is realistic on LC25000. Results vary by split/training budget.

## 🔧 Tech Stack
Python, PyTorch, torchvision, scikit-learn, Albumentations, OpenCV, Matplotlib, YAML

---

## 📚 Dataset: LC25000 (Lung & Colon Cancer Histopathology)
- 25,000 images, **5 classes** (5,000 each): `colon_aca`, `colon_n`, `lung_aca`, `lung_scc`, `lung_n`.
- Original layout after unzip (example):
```
lung_colon_image_set/
  colon_image_sets/{colon_aca, colon_n}/
  lung_image_sets/{lung_aca, lung_scc, lung_n}/
```
- Sources:
  - Kaggle (recommended): `andrewmvd/lung-and-colon-cancer-histopathological-images`
  - Paper: https://arxiv.org/abs/1912.12142
  - Maintainers' GitHub README

### ⚡ One-command download & prep
```bash
python scripts/download_lc25000.py --out data/raw
python scripts/prepare_lc25000.py --raw_dir data/raw/lung_colon_image_set --out_dir data/lc25000 --val_ratio 0.1 --test_ratio 0.1
```

> If you cannot use Kaggle API, download `LC25000.zip` from official mirrors and pass its unzipped folder to `--raw_dir`.

---

## 🚀 Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_lc25000.py --out data/raw
python scripts/prepare_lc25000.py --raw_dir data/raw/lung_colon_image_set --out_dir data/lc25000

python src/train.py --config config.yaml
python src/evaluate.py --config config.yaml
python src/infer.py --config config.yaml --image path/to/sample.jpg
```

---

## 🧪 Methodology
- **Augmentations** (Albumentations): flips, rotations, color jitter, random crops, optional Gaussian blur.
- **Model**: `resnet18` or `efficientnet_b0` (ImageNet pretrained), classifier head replaced for 5-way classification.
- **Optimization**: AdamW, cosine LR (or OneCycle), label smoothing CE, AMP (mixed precision).
- **Regularization**: weight decay, dropout, early stopping, optional EMA.
- **Cross-Validation**: stratified K-fold over file list (K configurable).

---

## 📂 Project Structure
```
biomed_image_classification_lc25000/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── download_lc25000.py
│   └── prepare_lc25000.py
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── infer.py
    └── utils.py
```

---

## 🧰 Tips to hit CE ≈ 0.42
- Use `efficientnet_b0`, image size 224–256, label smoothing 0.05, strong color/rotation jitter.
- Train ~20–30 epochs with cosine LR, early stopping patience ~5.
- Use K-fold CV and average the CE across folds to report.

---

## 📝 License & Data Use
Code: MIT. Dataset: follow data source license/terms (Kaggle/LC25000). This repo does **not** redistribute LC25000 due to licensing & size.
