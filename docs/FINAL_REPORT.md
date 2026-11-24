# Robust Hate Speech Detection in Romanized Punjabi-English Code-Mixed Social Media Text

**Minor Project — Final Report**

*Author: Palak · Report compiled 2026-11-23 · All numbers drawn from `results/metrics/`.*

---

## Abstract

This project investigates automatic hate-speech detection in **Romanized (Latin-script)
Punjabi-English code-mixed** social-media text — a low-resource setting for which no
public labelled dataset exists. We assemble a dataset from four public/collected
sources, build a reproducible pipeline, and compare a **BiLSTM + Attention** neural
baseline against three multilingual transformers (**MuRIL, XLM-RoBERTa, IndicBERTv2**)
across three training regimes. We evaluate not only on in-domain test data but on a
held-out set of **1,591 authentic YouTube comments** (the real yardstick).

**Headline finding:** transformers clearly beat the neural baseline *in-domain* and are
*far more robust to spelling noise*, but **every model — including transformers
fine-tuned on Romanized Punjabi — fails to detect authentic Punjabi hate** (macro-F1
≤ 0.51, hate-recall ≤ 40%). The bottleneck is **the scarcity of authentic, annotated
in-domain Punjabi hate examples (only 57 in the real test set)**, not the model, the
language, or the data volume. This is a reproducible, honest negative result that
motivates targeted data collection as the key future step.

---

## 1. Problem statement & objectives

Romanized Punjabi-English (e.g. *"veere tu bahut vadiya kaam kita"*) is how a large
share of Punjabi speakers actually write online, yet hate-speech resources for it are
effectively non-existent. The project objectives:

1. Assemble a usable Romanized Punjabi-English dataset.
2. Build a BiLSTM+Attention baseline and three transformer models.
3. **RQ1** — Compare neural vs transformer performance (in-domain and on authentic data).
4. **RQ2** — Measure robustness to spelling noise and code-mixing.
5. **RQ3** — Attempt a targeted improvement (adversarial data augmentation).
6. **RQ4** — Error analysis of difficult cases.

Binary task: `0 = not_hate`, `1 = hate`. Primary metric: **macro-F1** (robust to class
imbalance), with per-class precision/recall and hate-recall reported throughout.

---

## 2. Dataset sources & provenance

All data is consolidated in `More_datasets/combined_dataset.csv` (108,205 rows) and
processed into task-specific splits under `data/processed/`. Four sources were used:

| # | Source | Language / type | License & access | Role in project |
|---|--------|-----------------|------------------|-----------------|
| 1 | **PRISM Hinglish Hate Speech** (`pankajbiswas6/prism-hinglish-hate-speech`, HuggingFace) | Hindi-English code-mix, Latin script | MIT, direct download | Transfer-learning training corpus (Hinglish → Punjabi) |
| 2 | **IndicGuard — Punjabi** (AI4Bharat lineage) | Punjabi in Gurmukhi + romanized safety-prompt text | Public | Volume for the Romanized Punjabi training set (transliterated to Roman) |
| 3 | **Authentic YouTube comments** (collected via YouTube Data API v3) | Real Romanized Punjabi-English social-media text | Collected under official API, anonymised | **The authentic gold / real test set** |
| 4 | **Davidson et al. 2017** (English) | English hate/offensive | Public, research | Auxiliary English rows (not central to Punjabi evaluation) |

**Data-collection ethics:** authentic comments were gathered via the **official YouTube
Data API v3** (not scraping, no ToS violation). Only the comment **text** and
`video_id` are stored; all `@handles`/usernames are stripped — no personal data is
retained (`src/collect_youtube.py`).

### Processed splits actually used

| Split (file) | Rows | not_hate / hate | What it is |
|---|---|---|---|
| `train.csv` / `val.csv` / `test.csv` | 13,831 / 2,964 / 2,964 | balanced | PRISM `full` (all Latin rows) |
| `hinglish_train/val/test.csv` | 3,308 / 697 / 760 | — | PRISM Hinglish code-mix subset only |
| `rp_train.csv` / `rp_val.csv` | 18,295 / 3,921 | — | Romanized Punjabi (IndicGuard-translit) |
| `rp_test.csv` | 5,512 | 4,271 / 1,241 | Held-out RP + all authentic rows |
| **`rp_real_test.csv`** | **1,591** | **1,534 / 57** | **Authentic YouTube only — the honest yardstick** |
| `punjabi_gold_clean.csv` | 1,176 | 1,137 / 39 | Authentic Latin YouTube gold |
| `punjabi_seed_clean.csv` | 119 | 65 / 54 | Hand-authored synthetic dev seed |

> **Composition caveat (documented honestly):** of the 27,728-row Romanized Punjabi
> dataset, **26,137 rows (94%) are transliterated IndicGuard *safety-prompt* text**,
> and only **1,591 are authentic social-media comments** (1,176 natively Romanized +
> 415 Gurmukhi→Roman transliterated). Authentic rows are **held entirely out of
> train/val** and placed only in the test / real-test sets, so no authentic data leaks
> into training.

---

## 3. Exact pipeline (reproducible)

Every stage is a script in `src/`. Run top-to-bottom to reproduce all results.

```
┌─ STAGE 1 · Data acquisition ──────────────────────────────────────────────┐
│ src/data_loader.py        → download PRISM, standardise, Latin subset       │
│                             → data/processed/hinglish_all.csv (19,768 rows) │
│ src/collect_youtube.py    → authentic Punjabi comments via YouTube API v3   │
│                             (text + video_id only, @handles stripped)       │
└────────────────────────────────────────────────────────────────────────────┘
┌─ STAGE 2 · Preprocess & build datasets ───────────────────────────────────┐
│ src/preprocess.py         → clean_text(): URLs, mentions, emoji→text,       │
│                             elongation (bhoottt→bhoott), punctuation        │
│ src/make_splits.py        → stratified train/val/test (SEED=42, 70/15/15)   │
│ src/build_punjabi_gold.py → extract authentic youtube rows → gold set       │
│ src/build_romanized_punjabi_dataset.py                                      │
│     → IndicGuard Gurmukhi + YouTube Gurmukhi transliterated to Roman        │
│       (indic-transliteration, ITRANS scheme; strips [.~^|] artifacts)       │
│     → rp_train / rp_val / rp_test / rp_real_test (authentic held out)       │
│ src/generate_seed.py      → 119-row hand-authored Punjabi dev seed          │
│ src/eda.py                → corpus stats + figures                          │
└────────────────────────────────────────────────────────────────────────────┘
┌─ STAGE 3 · Model training ────────────────────────────────────────────────┐
│ src/models/bilstm_attention.py  BiLSTM + additive (Bahdanau) attention      │
│ src/train_bilstm.py       → baseline, LOCAL (Apple MPS)                      │
│                             variants: full | hinglish | rp                  │
│ src/train_transformer.py  → MuRIL / XLM-R / IndicBERTv2, on Colab GPU        │
│                             (HuggingFace Trainer; MAX_LEN=128, lr=2e-5,      │
│                              batch=16, 4 epochs)                             │
│   Colab round-trip: package_for_colab.py → notebook → import_colab_outputs.py│
└────────────────────────────────────────────────────────────────────────────┘
┌─ STAGE 4 · Evaluation & analysis ─────────────────────────────────────────┐
│ src/metrics.py            → macro-F1, per-class P/R/F1, appends comparison.csv│
│ src/robustness_eval.py    → 6 perturbations, macro-F1 degradation (RQ2)     │
│ src/train_augmented.py    → adversarial augmentation (RQ3)                   │
│ src/error_analysis.py     → FP/FN categorisation (RQ4)                       │
│ src/compare_results.py    → pivot tables + bar charts + RESULTS.md           │
│ src/make_report.py        → docs/RESULTS_REPORT.md                           │
└────────────────────────────────────────────────────────────────────────────┘
```

**Models:** BiLSTM+Attn (embed 200, hidden 256, 2 layers, dropout 0.3) ·
`google/muril-base-cased` · `xlm-roberta-base` · `ai4bharat/IndicBERTv2-MLM-only`.
**Compute:** BiLSTM trained locally on Apple MPS; transformers on a Google Colab T4 GPU.

---

## 4. Results

### RQ1 — Neural vs transformer

**(a) In-domain test (Hinglish) — macro-F1.** Transformers win, as expected:

| Model | full | hinglish | rp |
|---|---|---|---|
| BiLSTM+Attn | 0.752 | 0.653 | 0.716 |
| MuRIL | **0.796** | 0.682 | 0.708 |
| XLM-R | **0.797** | 0.694 | 0.692 |
| IndicBERTv2 | 0.784 | 0.699 | 0.701 |

**(b) Authentic real test (`rp_real_test`, 1,591 real YouTube rows / 57 hate) — the yardstick:**

| Model | train | macro-F1 | hate-F1 | hate-recall |
|---|---|---|---|---|
| IndicBERTv2 | hinglish | **0.509** | 0.095 | 0.21 |
| MuRIL | full | 0.498 | 0.099 | 0.28 |
| IndicBERTv2 | full | 0.494 | 0.095 | 0.28 |
| BiLSTM+Attn | hinglish | 0.488 | 0.057 | 0.12 |
| XLM-R | full | 0.471 | 0.095 | 0.39 |
| XLM-R | **rp** | 0.434 | 0.078 | 0.40 |
| IndicBERTv2 | **rp** | 0.426 | 0.079 | 0.44 |
| MuRIL | **rp** | 0.420 | 0.072 | 0.40 |
| BiLSTM+Attn | **rp** | 0.419 | 0.040 | 0.19 |

**Key finding:** on authentic data, **no model exceeds 0.51 macro-F1**, and hate-recall
never passes 0.44 — most models miss 60–80% of real hate. Critically, **transformers
trained on the Romanized Punjabi (`rp`) dataset do NOT beat simple Hinglish transfer**
(muril-rp 0.42 ≈ bilstm-rp 0.42). More Punjabi data of the *wrong domain*
(transliterated safety prompts) does not help.

### RQ2 — Robustness to spelling noise & code-mixing

macro-F1 under each perturbation (`full` models), and the combined-noise case (`all`):

| Perturbation | BiLSTM | MuRIL | XLM-R | IndicBERT |
|---|---|---|---|---|
| clean | 0.752 | 0.796 | 0.797 | 0.783 |
| char_elongation | 0.593 | 0.759 | 0.762 | 0.726 |
| char_swap | 0.649 | 0.760 | 0.780 | 0.739 |
| vowel_drop | 0.661 | 0.769 | 0.785 | 0.761 |
| leet_noise | 0.607 | 0.745 | 0.743 | 0.736 |
| spelling_variant | 0.753 | 0.797 | 0.797 | 0.784 |
| code_mix_inject | 0.752 | 0.796 | 0.797 | 0.784 |
| **all (combined)** | **0.370** | **0.627** | **0.562** | **0.567** |

**Finding:** the **word-level BiLSTM is highly vulnerable to character-level noise**
(−0.10 to −0.16 per perturbation; combined collapse to 0.37) because corrupted words
become out-of-vocabulary. **Transformers, using subword tokenization, are far more
robust** (combined case stays 0.56–0.63). Spelling-variant and code-mix injection barely
affect any model. This directly answers RQ2: subword models degrade gracefully under
the noise typical of real social media.

### RQ3 — Targeted improvement (adversarial augmentation)

Training the BiLSTM on perturbed copies of the data (char elongation/swap/vowel-drop/
leet/spelling), then re-measuring robustness:

| Perturbation | before | after | gain |
|---|---|---|---|
| clean | 0.752 | 0.749 | −0.004 |
| char_elongation | 0.593 | 0.712 | **+0.119** |
| char_swap | 0.649 | 0.720 | +0.070 |
| vowel_drop | 0.661 | 0.722 | +0.060 |
| leet_noise | 0.607 | 0.717 | **+0.110** |
| **all (combined)** | 0.370 | 0.613 | **+0.243** |

**Finding:** augmentation **substantially improves noise robustness** (combined-noise
case recovered from 0.37 → 0.61) at near-zero cost to clean accuracy — RQ3 answered
positively for *robustness*.

> ⚠️ **Honest caveat:** on the synthetic dev seed the augmented model jumped to 0.73,
> but on the **real Punjabi gold set it scored ~0.40 — no better than the plain model.**
> The synthetic gain was an artifact (the seed shared noise vocabulary with the
> augmentation). Lesson: augmentation improves in-domain robustness but does **not**
> transfer to authentic Punjabi hate — that needs real Punjabi training data.

### RQ4 — Error analysis

From `errors_bilstm_attn_full_indomain.csv`: **722 errors (250 false positives, 472
false negatives)**. Category breakdown of errors:

| Category | Count |
|---|---|
| negation / possible sarcasm | 221 |
| code-mixed | 34 |
| very short | 26 |
| character noise | 9 |
| profanity present | 8 |

**Finding:** errors are dominated by **missed hate (false negatives)** and by
**negation/sarcasm** constructions, where surface words don't match intent. On Punjabi,
failures are almost entirely a **hate-recall collapse** — the model defaults to
"not_hate". Uncertain cases are exported to `data/annotation/review_cases.csv` for
Punjabi-speaker review.

---

## 5. Live demonstration (why the headline matters)

Running `bilstm_attn_full` on five Punjabi examples:

| Prediction | Text | Correct? |
|---|---|---|
| not_hate (0.93) | *veere tu bahut vadiya kaam kita, shabash* (praise) | ✓ |
| not_hate (0.60) | *bai ji tuhada din vadiya rahe* (blessing) | ✓ |
| not_hate (0.73) | *tusi saare chutiye ho, maar dena chahida tuhanu* (threat) | ✗ |
| not_hate (0.68) | *eh log saare kutte ne, inha nu desh chado* (dehumanising) | ✗ |
| not_hate (0.64) | *tere jaisa gandda insaan… mar ja* (abuse) | ✗ |

The model reliably recognises **normal/benign** Punjabi but **misses clear hate** —
exactly the hate-recall collapse the metrics report.

---

## 6. Limitations & honest caveats

- **Authentic Punjabi hate is scarce.** Only **57 real hate examples** exist in the
  evaluation set; the training volume is transliterated *safety-prompt* text, not
  social media. This is the single dominant limitation.
- **PRISM English rows contain Wikipedia editing text** (out-of-domain), which inflates
  the `full` in-domain scores (see `results/metrics/eda_summary.txt`).
- **Augmentation's Punjabi gain is confounded** (RQ3 caveat).
- **Transliteration is imperfect** — Gurmukhi→Roman (ITRANS) does not perfectly match
  how people naturally romanize Punjabi.

---

## 7. Conclusion & future work

The project delivers a complete, reproducible pipeline and a comparison of four models
across three regimes, evaluated honestly on authentic data. The central, defensible
contribution is a **negative result**: *state-of-the-art multilingual models and
transfer learning still fail at authentic low-resource Romanized Punjabi hate detection,
and the binding constraint is authentic annotated in-domain data — not model capacity,
language coverage, or data volume.* Secondary contributions: transformers are
substantially more noise-robust than the word-level baseline (RQ2), and adversarial
augmentation recovers robustness in-domain (RQ3).

**Future work (highest-impact first):**
1. **Collect + annotate a few hundred–thousand authentic Romanized Punjabi hate
   comments** (pipeline already built: `collect_youtube.py` → `make_annotation_sheet.py`
   → Cohen's-κ agreement → `agreement.py`).
2. **Two-stage adaptation** (`train_adapt.py`): Hinglish → authentic Punjabi fine-tuning
   once a real gold set exists.
3. Re-run the full comparison on the enlarged authentic set.

---

## Appendix — Artifacts

- **Metrics:** `results/metrics/comparison.csv` (51 rows: 5 models × 3 variants × 4 eval sets),
  per-model JSONs, robustness CSVs.
- **Figures:** `results/figures/` — `compare_indomain.png`, `compare_transfer.png`,
  9 `robustness_*.png`, EDA figures.
- **Auto-generated report:** `docs/RESULTS_REPORT.md` (refresh via `python src/make_report.py`).
- **Plan & guidelines:** `docs/PROJECT_PLAN.md`, `docs/ANNOTATION_GUIDELINES.md`.
