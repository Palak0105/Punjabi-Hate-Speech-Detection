"""Central configuration for the Punjabi-English hate speech detection project.

Edit paths and hyper-parameters here; every other module imports from this file
so experiments stay reproducible.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded datasets, untouched
EXTERNAL_DIR = DATA_DIR / "external"  # third-party corpora
PROC_DIR = DATA_DIR / "processed"     # cleaned, split, ready-to-train
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRIC_DIR = RESULTS_DIR / "metrics"

for _d in (RAW_DIR, EXTERNAL_DIR, PROC_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Labels  (binary task first; extend to 3-class later if data supports it)
# ---------------------------------------------------------------------------
LABELS = ["not_hate", "hate"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# Canonical column names used across the project after preprocessing
TEXT_COL = "text"
CLEAN_COL = "clean_text"
LABEL_COL = "label"      # integer id
SPLIT_COL = "split"      # train / val / test

# ---------------------------------------------------------------------------
# Splits & reproducibility
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # fraction of the (train+val) remainder

# ---------------------------------------------------------------------------
# Transformer model checkpoints (HuggingFace ids)
# ---------------------------------------------------------------------------
TRANSFORMERS = {
    "muril": "google/muril-base-cased",
    "xlmr": "xlm-roberta-base",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
}

MAX_LEN = 128
TRAIN_BATCH = 16
EVAL_BATCH = 32
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# ---------------------------------------------------------------------------
# BiLSTM baseline
# ---------------------------------------------------------------------------
BILSTM = {
    "vocab_size": 30000,
    "embed_dim": 200,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "max_len": 60,
    "batch_size": 64,
    "epochs": 15,
    "lr": 1e-3,
}
