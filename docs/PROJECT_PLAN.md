# Build Plan — Robust Hate Speech Detection in Romanized Punjabi-English Code-Mixed Text

This is the working engineering plan that turns the synopsis into runnable code.
It is organised as **stages**; each stage has concrete deliverables, the file(s)
it produces, and the synopsis objective it satisfies.

## Data strategy (decided)

No public Romanized Punjabi-English hate-speech dataset exists (verified: HuggingFace,
arXiv, GitHub — only Punjabi ASR/TTS/NER/translation sets, and *song* toxicity).

**Candidate datasets investigated (Nov 2026):**
| Resource | Verdict |
|----------|---------|
| **PRISM** (HF `pankajbiswas6/prism-hinglish-hate-speech`) | ✅ **CHOSEN** — binary Hinglish hate, MIT, directly downloadable. Latin subset = 19,768 rows (4,774 Romanized hinglish + 14,994 English). |
| PGT-FG (Punjabi songs, Zenodo) | ❌ Labels released **without lyric text** — unusable for a text classifier. |
| IndicSafe (HF, Punjabi `pa`) | ❌ Gurmukhi-script LLM-safety prompts, not social media. |
| HINGHATE (IEEE DataPort) | ⏳ Download gated behind account login — optional backup. |
| P-HSSN / ACM 3677176 | 📄 Paywalled; Gurmukhi — literature-review citations only. |

We therefore use a **hybrid** design:

1. **Primary train/dev corpus** — a public **Hindi-English code-mixed** hate/offensive
   dataset (Romanized, social-media, labelled). This is the linguistically closest
   large resource and lets us train BiLSTM + all three transformers properly.
2. **Punjabi-English evaluation set** — a **small, hand-built (~500–1000)** Romanized
   Punjabi-English test set. Used for the difficult-case analysis, robustness tests,
   and human validation. This is where the genuine Punjabi contribution lives.

### Why Hindi-English (Hinglish) data is useful for a Punjabi task
Punjabi and Hindi are both Indo-Aryan; Romanized on social media they share
spelling conventions (`nahi, hai, tu, yaar`), much vocabulary, code-mixing grammar,
slurs, and informal register. So Hinglish data serves three roles:
1. **Teaches the task** — it is the only large labelled code-mixed Indic hate corpus;
   it provides the examples needed to learn hate vs not-hate in the first place.
2. **Transfer baseline (a finding)** — "how far does Hindi→Punjabi transfer reach?"
   Measured on the BiLSTM: macro-F1 0.75 (in-domain) → 0.44 (Punjabi), hate recall
   11%. Quantifies the cross-lingual gap.
3. **Base for two-stage adaptation** — fine-tune on Hinglish (task + volume), then
   further fine-tune on the small Punjabi gold set (language + domain fit). Standard
   low-resource recipe; expected best model. Implemented in `src/train_adapt.py`.

Transformers (MuRIL, IndicBERTv2, XLM-R) are pretrained on Punjabi *and* Hindi, so
they should transfer better than the from-scratch BiLSTM — this contrast is RQ1.
**Limitation:** Hinglish alone does NOT give strong Punjabi performance; the real
Punjabi gold set is required for the final results.

This keeps every synopsis objective achievable within the timeline while being
honest about what the models were trained on (a transfer/generalisation framing —
which is itself a legitimate research finding for a low-resource language).

> **Labelling scheme (binary to start):** `hate` vs `not_hate`.
> Define `hate` = content attacking a person/group by identity, or abusive/
> threatening language targeting them. Keep a written guideline doc; for any
> self-annotated Punjabi data use **2 annotators** and report **Cohen's κ**.

---

## Stages

### Stage 0 — Environment & scaffolding  ✅ (done)
- `requirements.txt`, `.gitignore`, folder structure, `src/config.py`.
- `src/preprocess.py` — Romanized Punjabi-English cleaning (2 levels).
- `src/perturb.py` — robustness perturbation engine.
- **Objective:** 2 (preprocessing), foundation for all.

### Stage 1 — Acquire & standardise the Hindi-English corpus  ✅ (done)
- `src/data_loader.py`: downloads PRISM, filters to Latin script, maps to binary
  scheme, outputs tidy CSV `text, label, lang, source, orig_split`.
- `--lang-mode`: `hinglish` (romanized only, 4,774) | `latin` (default, 19,768) | `all`.
- **Deliverable:** `data/processed/hinglish_all.csv` — 19,768 rows, 10,406 not_hate /
  9,362 hate. Raw CSVs cached in `data/raw/`.
- **Objective:** 1.

### Stage 1b — Build the Punjabi-English evaluation set (from scratch)  🔨 in progress
The genuine research contribution. Real YouTube-API data, human-annotated.
Pipeline (all scripts built & tested):
1. **Collect** — `src/collect_youtube.py`: pulls comments from Punjabi videos via
   the official YouTube Data API (needs a free `YOUTUBE_API_KEY`). Stores only
   text + video id; strips @handles/URLs; heuristic Romanized-Punjabi filter.
2. **Make sheets** — `src/make_annotation_sheet.py`: raw comments → two blank
   annotator sheets (`annotation_A/B.csv`) with stable ids.
3. **Annotate** — two people label independently per `docs/ANNOTATION_GUIDELINES.md`
   (binary hate/not_hate + secondary fields).
4. **Agreement** — `src/agreement.py`: Cohen's κ + exports disagreements/uncertain
   to `adjudication.csv`.
5. **Adjudicate → gold** — fill `gold_label`, rerun with `--adjudicated` →
   `data/processed/punjabi_gold.csv`.
- **Target size:** ~500–1000 gold rows. **Objective:** 1, 8.
- **Blocker:** real gold set needs a YouTube Data API key (free) from the user.
- **Interim:** `src/generate_seed.py` → `data/processed/punjabi_seed.csv` (120
  author-generated dev rows). Lets Stages 2–6 be built/tested now. **Not** for
  final results — `punjabi_gold.csv` (real, annotated) replaces it. Rows tagged
  `source=seed_synth` so synthetic never silently mixes with real data.

### Stage 2 — Preprocess & split
- `src/make_splits.py`: apply `Preprocessor`, create `clean_text`, stratified
  train/val/test split (seeded), save to `data/processed/{train,val,test}.csv`.
- EDA notebook `notebooks/01_eda.ipynb`: class balance, length distribution,
  code-mix ratio estimate, most frequent tokens per class.
- **Deliverable:** split CSVs + EDA figures.
- **Objective:** 1, 2, 9 (class distribution).

### Stage 3 — BiLSTM + Attention baseline
- `src/models/bilstm_attention.py`: embedding → BiLSTM → additive attention →
  classifier. `src/train_bilstm.py` train/eval loop with early stopping.
- Runs on Mac (MPS) or Colab.
- **Deliverable:** trained baseline + metrics row in `results/metrics/`.
- **Objective:** 3.

### Stage 4 — Transformer fine-tuning (Colab GPU)
- `src/train_transformer.py`: generic HuggingFace `Trainer` wrapper parameterised
  by model key (`muril`, `xlmr`, `indicbert`). One notebook `notebooks/02_transformers.ipynb`
  runs all three on Colab.
- Save best checkpoint + per-model metrics.
- **Objective:** 4.

### Stage 5 — Evaluation & comparison
- `src/evaluate.py`: accuracy, precision, recall, F1, macro-F1, confusion matrix;
  aggregate all models into one comparison table + bar chart.
- **Deliverable:** `results/metrics/comparison.csv`, `results/figures/*.png`.
- **Objective:** 9.

### Stage 6 — Robustness / difficult-case analysis  ⭐ (the differentiator)
- `src/robustness_eval.py`: for each trained model, evaluate on clean test set and
  on each perturbation from `perturb.py`; compute degradation + robustness score.
- Apply the Punjabi-English eval set here too.
- **Deliverable:** robustness table + heatmap (model × perturbation).
- **Objective:** 5, answers RQ2.

### Stage 7 — Error analysis + human validation
- `notebooks/03_error_analysis.ipynb`: inspect false positives / false negatives,
  categorise (sarcasm, spelling, code-mix, ambiguous). Export ambiguous cases to a
  sheet for Punjabi-speaking reviewers; log agreement.
- **Objective:** 6, 8.

### Stage 8 — Targeted improvement
Two complementary techniques (pick based on observed errors):
- **Two-stage adaptation** — `src/train_adapt.py`: fine-tune the best Hinglish model
  further on the Punjabi gold set (Hinglish→Punjabi). Expected to close most of the
  transfer gap. Needs `punjabi_gold.csv`.
- **Adversarial/augmented training** — add perturbed copies of train data (via
  `perturb.py`) to improve robustness. Retrain, re-run Stage 6, report the gain.
- **Objective:** 7, answers RQ3.

### Stage 9 — Final report & figures
- Collate tables/figures; write results into the report; finalise README.
- **Objective:** final evaluation.

---

## Research questions → where answered
| RQ | Stage |
|----|-------|
| RQ1 neural vs transformer comparison | 3, 4, 5 |
| RQ2 effect of spelling/code-mix | 6 |
| RQ3 can improvements raise robustness | 8 |
| RQ4 performance across classes / hard cases | 5, 7 |

## Immediate next action
Stage 1 — lock down the Hindi-English source file and write `data_loader.py`.
