"""Unpack Colab outputs (colab_outputs.zip) back into the workspace.

The notebook produces colab_outputs.zip containing project/results/... and
project/models/... . Download it, drop it in the project root, then run this to
merge it in. comparison.csv is MERGED + de-duplicated (keeping the latest row per
model/variant/eval), so re-runs don't create duplicate rows.

Run:
    python src/import_colab_outputs.py                       # expects ./colab_outputs.zip
    python src/import_colab_outputs.py --zip ~/Downloads/colab_outputs.zip
    python src/import_colab_outputs.py --zip ... --no-models  # skip big checkpoints
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def _merge_comparison(incoming: Path) -> None:
    local = C.METRIC_DIR / "comparison.csv"
    new = pd.read_csv(incoming)
    if local.exists():
        combined = pd.concat([pd.read_csv(local), new], ignore_index=True)
    else:
        combined = new
    key = ["model", "train_variant", "eval_on"]
    combined = combined.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    combined.to_csv(local, index=False)
    print(f"  merged comparison.csv -> {len(combined)} unique rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=C.ROOT / "colab_outputs.zip")
    ap.add_argument("--no-models", action="store_true", help="skip model checkpoints")
    args = ap.parse_args()

    if not args.zip.exists():
        sys.exit(f"not found: {args.zip}\nDownload colab_outputs.zip from the notebook first.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(tmp)
        base = tmp / "project"
        if not base.exists():
            base = tmp  # in case zip has no project/ prefix

        # 1) metrics json files
        src_metrics = base / "results" / "metrics"
        copied = 0
        if src_metrics.exists():
            for jf in src_metrics.glob("*.json"):
                shutil.copy2(jf, C.METRIC_DIR / jf.name)
                copied += 1
            comp = src_metrics / "comparison.csv"
            if comp.exists():
                _merge_comparison(comp)
        print(f"  copied {copied} metric json files")

        # 2) model checkpoints
        if not args.no_models:
            src_models = base / "models"
            if src_models.exists():
                for md in src_models.iterdir():
                    if md.is_dir():
                        dest = C.MODEL_DIR / md.name
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(md, dest)
                        print(f"  imported model: {md.name}")
        else:
            print("  skipped model checkpoints (--no-models)")

    print("\nDone. Now run Stage 5 (comparison) locally.")


if __name__ == "__main__":
    main()
