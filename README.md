# Robust Hate Speech Detection in Romanized Punjabi-English Code-Mixed Social Media Text

Minor project — detecting hate speech in Romanized Punjabi-English code-mixed
social-media text, with a focus on **robustness** to spelling variation, noise,
and code-mixing.

Models compared: **BiLSTM + Attention** (neural baseline), **MuRIL**,
**XLM-RoBERTa**, **IndicBERTv2**.

## Project structure
```
.
├── data/
│   ├── raw/          # downloaded datasets (git-ignored)
│   ├── external/     # third-party corpora
│   └── processed/    # cleaned + split, ready to train
├── src/
│   ├── config.py     # all paths & hyper-parameters
│   ├── preprocess.py # Romanized Punjabi-English text cleaning
│   └── perturb.py    # robustness perturbation engine
├── notebooks/        # EDA, training (Colab), analysis
├── models/           # saved checkpoints (git-ignored)
├── results/          # metrics + figures
└── docs/
    └── PROJECT_PLAN.md   # the staged build plan — read this
```

## Setup
System Python may block installs (PEP 668), so use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Transformer fine-tuning runs on **Google Colab / Kaggle** (free GPU); torch is
preinstalled there. Local Mac can run the BiLSTM baseline via Apple MPS.

## Data
No public Romanized Punjabi-English hate-speech dataset exists. We use a **hybrid**
strategy — train on a public Hindi-English code-mixed hate corpus, evaluate on a
small hand-built Punjabi-English set. See `docs/PROJECT_PLAN.md` for details.

## Stage 4 — transformers on Colab (round-trip)
```bash
# 1. laptop: bundle code + data
python src/package_for_colab.py          # -> colab_bundle.zip

# 2. Colab (GPU runtime): open notebooks/02_transformers_colab.ipynb,
#    upload colab_bundle.zip, run all cells -> downloads colab_outputs.zip

# 3. laptop: import trained models + metrics back
python src/import_colab_outputs.py       # expects ./colab_outputs.zip
```
Everything lands in `results/metrics/comparison.csv` alongside the BiLSTM rows.

## Quick sanity check
```bash
python3 src/preprocess.py   # demo cleaning
python3 src/perturb.py      # demo robustness perturbations
```

## Status
- [x] Stage 0 — scaffolding, preprocessing, perturbations
- [x] Stage 1 — acquire Hindi-English corpus (PRISM, 19,768 rows)
- [x] Stage 1b — REAL Punjabi gold set built from your YouTube data
      (punjabi_gold.csv, 1176 rows / 39 hate); synthetic seed retained for reference
- [x] Stage 2 — preprocess & split + EDA (⚠ found Wikipedia contamination in
      PRISM English not-hate rows; hinglish subset is clean — see eda_summary.txt)
- [x] Stage 3 — BiLSTM+Attention baseline (full vs hinglish; in-domain macro-F1
      0.75/0.65, Punjabi transfer 0.44/0.39 — large transfer gap confirmed)
- [x] Stage 4 — transformer fine-tuning (MuRIL/XLM-R/IndicBERT, both variants)
- [x] Stage 5 — comparison (transformers win in-domain ~0.80; all transfer poorly to Punjabi)
- [x] Romanized Punjabi dataset built (27.7k) + BiLSTM trained; in-dataset 0.72 but authentic real test 0.42 (IndicGuard domain gap)
- [~] Stage 6 — robustness (BiLSTM done locally; transformers run on next Colab)
- [~] Stage 7 — error analysis done (BiLSTM); review_cases.csv ready for Punjabi reviewer
- [x] Stage 8 — augmented training (robustness +0.085 mean, combined-noise +0.24; 2-stage adaptation awaits gold set)
- [~] Stage 9 — report scaffold auto-generated (docs/RESULTS_REPORT.md); final write-up after gold set
