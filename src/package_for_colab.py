"""Bundle the code + data Colab needs into one zip to upload.

Creates colab_bundle.zip in the project root, preserving the project/ layout so
config.py's relative paths work unchanged on Colab.

Run:
    python src/package_for_colab.py
Then upload colab_bundle.zip in the notebook's upload cell.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# Files Colab needs (relative to project root). train_transformer imports only
# config + metrics, so we ship those plus the processed data + current results.
CODE = [
    "src/config.py", "src/metrics.py", "src/train_transformer.py",
    "src/perturb.py", "src/robustness_eval.py",   # for on-Colab robustness (Stage 6)
    "src/train_adapt.py",                          # for two-stage adaptation (Stage 8)
    "src/error_analysis.py",                       # for on-Colab error analysis (Stage 7)
]
DATA = [
    "data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv",
    "data/processed/hinglish_train.csv", "data/processed/hinglish_val.csv",
    "data/processed/hinglish_test.csv", "data/processed/punjabi_seed_clean.csv",
    "data/processed/punjabi_gold_clean.csv",  # REAL Punjabi gold (youtube)
    # Romanized Punjabi dataset (project core) + authentic real test
    "data/processed/rp_train.csv", "data/processed/rp_val.csv",
    "data/processed/rp_test.csv", "data/processed/rp_real_test.csv",
]
RESULTS = ["results/metrics/comparison.csv"]  # so transformer rows append to it


def main():
    root = C.ROOT
    out = root / "colab_bundle.zip"
    missing, added = [], 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in CODE + DATA + RESULTS:
            p = root / rel
            if p.exists():
                z.write(p, arcname=f"project/{rel}")
                added += 1
            else:
                missing.append(rel)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({added} files, {size_mb:.2f} MB)")
    if missing:
        print("  WARNING missing (run earlier stages first):")
        for m in missing:
            print("   -", m)
    print("\nUpload this zip in the notebook's step-2 cell.")


if __name__ == "__main__":
    main()
