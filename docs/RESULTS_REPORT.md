# Results Report — Robust Hate Speech Detection in Romanized Punjabi-English Code-Mixed Text

*Auto-generated from `results/metrics/` on 2026-11-23. Re-run `python src/make_report.py` to refresh.*

## 1. Methodology snapshot
- **Task:** binary hate / not_hate classification of Romanized code-mixed text.
- **Training data:** PRISM Hindi-English (Hinglish) hate corpus, Latin-script subset (19,768 rows). Two training variants: `full` (all Latin rows) and `hinglish` (Romanized code-mix only). See `docs/PROJECT_PLAN.md` for why Hinglish transfers to Punjabi.
- **Punjabi evaluation:** a 119-row author-generated *development seed* (`punjabi_seed`). The real human-annotated gold set is pending a YouTube-API collection (pipeline built: `collect_youtube.py` → `make_annotation_sheet.py` → `agreement.py`).
- **Models:** BiLSTM+Attention (neural baseline) vs MuRIL, XLM-R, IndicBERTv2 (transformers).
- **Metrics:** accuracy, macro-F1, per-class P/R/F1, confusion matrix.

## 2. RQ1 — Neural vs transformer comparison

**In-domain test (Hinglish) — macro-F1:**

|             |   full |   hinglish |       rp |
|:------------|-------:|-----------:|---------:|
| BiLSTM+Attn | 0.7524 |     0.6534 |   0.7165 |
| MuRIL       | 0.7962 |     0.6816 |   0.7079 |
| XLM-R       | 0.797  |     0.6944 |   0.6915 |
| IndicBERTv2 | 0.7845 |     0.699  |   0.7006 |
| BiLSTM+Aug  | 0.7485 |   nan      | nan      |

**Authentic real test (`rp_real_test`, 1,591 real YouTube rows / 57 hate) — the yardstick:**

|    | model       | train_variant   |   macro_f1 |   hate_f1 |   hate_recall |
|---:|:------------|:----------------|-----------:|----------:|--------------:|
|  0 | BiLSTM+Attn | full            |     0.4341 |    0.0343 |        0.1404 |
|  1 | BiLSTM+Attn | hinglish        |     0.4883 |    0.0565 |        0.1228 |
|  2 | BiLSTM+Aug  | full            |     0.3921 |    0.0608 |        0.386  |
|  3 | BiLSTM+Attn | rp              |     0.4186 |    0.0397 |        0.193  |
|  4 | MuRIL       | rp              |     0.42   |    0.0722 |        0.4035 |
|  5 | MuRIL       | full            |     0.4983 |    0.0988 |        0.2807 |
|  6 | MuRIL       | hinglish        |     0.4609 |    0.0864 |        0.3684 |
|  7 | XLM-R       | rp              |     0.4343 |    0.0781 |        0.4035 |
|  8 | XLM-R       | full            |     0.4711 |    0.0954 |        0.386  |
|  9 | XLM-R       | hinglish        |     0.4354 |    0.1037 |        0.5965 |
| 10 | IndicBERTv2 | rp              |     0.4257 |    0.0792 |        0.4386 |
| 11 | IndicBERTv2 | full            |     0.4942 |    0.0952 |        0.2807 |
| 12 | IndicBERTv2 | hinglish        |     0.5087 |    0.0952 |        0.2105 |

**Key finding:** training on the Romanized Punjabi dataset (94% transliterated IndicGuard *safety-prompt* text) reaches ~0.72 macro-F1 *in-dataset* but only ~0.42 on authentic data — **no better than Hinglish transfer**. More data of the wrong domain does not help. The bottleneck is authentic in-domain annotated Punjabi, not volume or language.

**REAL Punjabi gold (YouTube Latin only) — macro-F1:**

|             |   full |   hinglish |       rp |
|:------------|-------:|-----------:|---------:|
| BiLSTM+Attn | 0.4156 |     0.4892 |   0.4    |
| MuRIL       | 0.4915 |     0.4691 |   0.4191 |
| XLM-R       | 0.4905 |     0.467  |   0.4288 |
| IndicBERTv2 | 0.4849 |     0.5076 |   0.434  |
| BiLSTM+Aug  | 0.3978 |   nan      | nan      |

**Punjabi synthetic seed (transfer) — macro-F1:**  *(dev seed only; see caveats)*

|             |   full |   hinglish |       rp |
|:------------|-------:|-----------:|---------:|
| BiLSTM+Attn | 0.4391 |     0.3938 | nan      |
| MuRIL       | 0.482  |     0.3533 |   0.4538 |
| XLM-R       | 0.4547 |     0.3734 |   0.3569 |
| IndicBERTv2 | 0.4547 |     0.3533 |   0.4131 |
| BiLSTM+Aug  | 0.7264 |   nan      | nan      |

**Finding:** transformers outperform the BiLSTM baseline in-domain (best ~0.80 vs 0.75 macro-F1). All models transfer poorly to Punjabi (macro-F1 ~0.35–0.48, hate recall ≤13%): multilingual pretraining alone does not solve Punjabi hate detection. `full` beats `hinglish-only` everywhere (data volume wins).

## 3. RQ2 — Robustness to spelling noise & code-mixing

**BiLSTM (full) macro-F1 under perturbation:**

|    | perturbation     |   macro_f1 |   degradation |   robustness_ratio |
|---:|:-----------------|-----------:|--------------:|-------------------:|
|  0 | clean            |     0.7524 |        0      |             1      |
|  1 | char_elongation  |     0.5927 |        0.1597 |             0.7878 |
|  2 | char_swap        |     0.6494 |        0.103  |             0.8632 |
|  3 | vowel_drop       |     0.6612 |        0.0912 |             0.8788 |
|  4 | spelling_variant |     0.7527 |       -0.0004 |             1.0005 |
|  5 | leet_noise       |     0.6068 |        0.1456 |             0.8065 |
|  6 | code_mix_inject  |     0.7524 |        0      |             1      |
|  7 | all              |     0.3698 |        0.3825 |             0.4916 |

**Finding:** the word-level BiLSTM is most vulnerable to **character-level noise** (elongation, leet, swaps: −0.10 to −0.16 macro-F1 each) because corrupted words become out-of-vocabulary. Spelling-variant and code-mix injection barely hurt. Transformer robustness (subword tokenization — expected to be more robust) runs on the next Colab session (`robustness_eval.py`, wired into the notebook).

## 4. RQ3 — Targeted improvement (adversarial augmentation)

**Robustness before vs after augmented training (BiLSTM, full):**

|    | perturbation     |   before |   after |    gain |
|---:|:-----------------|---------:|--------:|--------:|
|  0 | clean            |   0.7524 |  0.7485 | -0.0039 |
|  1 | char_elongation  |   0.5927 |  0.712  |  0.1193 |
|  2 | char_swap        |   0.6494 |  0.7195 |  0.0701 |
|  3 | vowel_drop       |   0.6612 |  0.7216 |  0.0604 |
|  4 | spelling_variant |   0.7527 |  0.7485 | -0.0042 |
|  5 | leet_noise       |   0.6068 |  0.7171 |  0.1103 |
|  6 | code_mix_inject  |   0.7524 |  0.7485 | -0.0039 |
|  7 | all              |   0.3698 |  0.6126 |  0.2428 |

**Finding:** augmenting training data with perturbed copies improves noise robustness substantially (mean +0.085 macro-F1; combined-noise case recovered from ~0.37 to ~0.61) at near-zero cost to clean accuracy. This answers RQ3: targeted augmentation enhances robustness.

> ⚠️ **Caveat (now confirmed on real data):** the augmented model's *synthetic-seed* score jumped to 0.73, but on the **REAL Punjabi gold set it scores ~0.40 — no better (slightly worse) than the plain model**. The synthetic gain was an artifact (shared noise vocabulary). Lesson: augmentation improves in-domain *robustness* but does NOT transfer to real Punjabi hate; that needs Punjabi training data (two-stage adaptation).

## 5. RQ4 — Error analysis & difficult cases

From `errors_bilstm_attn_full_indomain.csv`: 722 errors (250 false positives, 472 false negatives).

**Error categories:**

|                        |   count |
|:-----------------------|--------:|
| other                  |     450 |
| negation_maybe_sarcasm |     221 |
| code_mixed             |      34 |
| very_short             |      26 |
| char_noise             |       9 |
| profanity_present      |       8 |

**Finding:** errors are dominated by **false negatives (missed hate)** and by **negation/sarcasm** constructions. The model is *confidently wrong* (mean confidence ~0.71 on errors). On Punjabi, failures are almost entirely missed hate (recall collapse). Uncertain cases exported to `data/annotation/review_cases.csv` for Punjabi-speaker review (Objective 8).

## 6. Limitations & honest caveats
- **No real Punjabi training/eval data yet.** Results use a synthetic dev seed; the human-annotated gold set is required for trustworthy Punjabi numbers.
- **PRISM English rows are partly Wikipedia editing text** (out-of-domain), which inflates the `full` in-domain scores. See `results/metrics/eda_summary.txt`.
- **Augmentation Punjabi gain is confounded** (see RQ3 caveat).
- **Transformer robustness & two-stage adaptation** results are pending the next GPU run / gold set.

## 7. Remaining work (gated on the Punjabi gold set)
1. Collect + annotate the real Romanized Punjabi-English gold set (κ-validated).
2. Re-run comparison, robustness, and error analysis on real Punjabi data.
3. Two-stage adaptation (`train_adapt.py`): Hinglish → Punjabi fine-tuning.
4. Final report write-up.

## 8. Figures (`results/figures/`)
- `augmentation_robustness.png`
- `bilstm_comparison.png`
- `class_balance.png`
- `compare_indomain.png`
- `compare_transfer.png`
- `length_by_class.png`
- `robustness_bilstm_attn_full.png`
- `robustness_bilstm_attn_hinglish.png`
- `top_hate_tokens.png`