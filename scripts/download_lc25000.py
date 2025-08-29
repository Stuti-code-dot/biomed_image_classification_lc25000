#!/usr/bin/env python
"""
Download LC25000 from Kaggle via the Kaggle API.
- Requires: kaggle account + API token placed at ~/.kaggle/kaggle.json
- Dataset slug: andrewmvd/lung-and-colon-cancer-histopathological-images
"""
import argparse, subprocess, sys
from pathlib import Path

def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = "andrewmvd/lung-and-colon-cancer-histopathological-images"
    zip_path = out / "lung-and-colon-cancer-histopathological-images.zip"
    try:
        print(f"Downloading {slug} to {zip_path} ...")
        subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", str(out), "-f", "lung-and-colon-cancer-histopathological-images.zip", "-q"])
        print("Unzipping...")
        subprocess.check_call(["unzip", "-q", "-o", str(zip_path), "-d", str(out)])
        print("Done. Raw data at:", out)
        print("Next: run scripts/prepare_lc25000.py to split into train/val/test.")
    except FileNotFoundError:
        print("Error: Kaggle CLI not found. Install with `pip install kaggle` and place API token in ~/.kaggle/kaggle.json")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed. Ensure you've accepted dataset terms on Kaggle & your API token is valid.")
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw", help="Output directory for raw download")
    args = ap.parse_args()
    main(args.out)
