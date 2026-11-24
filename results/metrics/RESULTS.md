# Results — model comparison

## In-domain test (Hinglish) — macro-F1
| model       |   full |   hinglish |     rp |
|:------------|-------:|-----------:|-------:|
| bilstm_attn | 0.7524 |     0.6534 | 0.7165 |
| muril       | 0.7962 |     0.6816 | 0.7079 |
| xlmr        | 0.797  |     0.6944 | 0.6915 |
| indicbert   | 0.7845 |     0.699  | 0.7006 |

## Punjabi transfer — macro-F1
| model       |   full |   hinglish |       rp |
|:------------|-------:|-----------:|---------:|
| bilstm_attn | 0.4391 |     0.3938 | nan      |
| muril       | 0.482  |     0.3533 |   0.4538 |
| xlmr        | 0.4547 |     0.3734 |   0.3569 |
| indicbert   | 0.4547 |     0.3533 |   0.4131 |

## Hate-class recall (full variant)
| model       |   indomain_test |   punjabi_gold |   punjabi_seed |   rp_real_test |
|:------------|----------------:|---------------:|---------------:|---------------:|
| bilstm_attn |           0.664 |          0.205 |          0.111 |          0.14  |
| muril       |           0.766 |          0.282 |          0.13  |          0.281 |
| xlmr        |           0.705 |          0.359 |          0.111 |          0.386 |
| indicbert   |           0.709 |          0.308 |          0.111 |          0.281 |

## Takeaways
- Best in-domain (full): **XLM-R** (0.797 macro-F1).
- Best Punjabi transfer (full): **MuRIL** (0.482 macro-F1).
- Transformers beat the BiLSTM baseline in-domain by 0.045 macro-F1.
- ALL models transfer poorly to Punjabi (macro-F1 ~0.35–0.48, hate recall ≤13%): Hinglish training alone does not solve Punjabi hate detection → motivates the real gold set + two-stage adaptation (train_adapt.py).
- 'full' beats 'hinglish-only' everywhere: more data wins despite the English/Wikipedia dilution.