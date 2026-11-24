"""Streamlit demo + results dashboard for the Romanized Punjabi hate-speech project.

Run locally:   streamlit run app/app.py
Deploy:        Streamlit Community Cloud -> main file path = app/app.py
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.bilstm_attention import BiLSTMAttention  # noqa: E402
from preprocess import clean_text  # noqa: E402

MODEL_DIR = ROOT / "models"
METRIC_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
LABELS = ["not_hate", "hate"]

MODELS = {
    "BiLSTM+Attn - trained on Hinglish (full)": "bilstm_attn_full.pt",
    "BiLSTM+Attn - trained on Hinglish code-mix only": "bilstm_attn_hinglish.pt",
    "BiLSTM+Attn - trained on Romanized Punjabi (rp)": "bilstm_attn_rp.pt",
    "BiLSTM+Attn - adversarially augmented (full)": "bilstm_attn_full_aug.pt",
}

EXAMPLES = [
    "veere tu bahut vadiya kaam kita, shabash",
    "bai ji tuhada din vadiya rahe",
    "tusi saare chutiye ho, maar dena chahida tuhanu",
    "eh log saare kutte ne, inha nu desh chado",
    "tere jaisa gandda insaan maine nahi dekhya, mar ja",
]


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def encode(text, vocab, max_len):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:max_len]
    return ids or [1]


@st.cache_resource(show_spinner="Loading model...")
def load_model(fname: str):
    ck = torch.load(MODEL_DIR / fname, map_location="cpu", weights_only=False)
    vocab, cfg = ck["vocab"], ck["config"]
    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=2, dropout=cfg["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, vocab, cfg


def predict(text: str, fname: str):
    model, vocab, cfg = load_model(fname)
    cleaned = clean_text(text, level="transformer", emoji_mode="demojize")
    toks = tokenize(cleaned)[: cfg["max_len"]]
    ids = encode(cleaned, vocab, cfg["max_len"])
    with torch.no_grad():
        logits, weights = model(torch.tensor([ids]), return_attention=True)
        probs = torch.softmax(logits, 1)[0].tolist()
    w = weights[0].tolist()[: len(toks)]
    return cleaned, toks, probs, w


def highlight(tokens, weights):
    if not tokens:
        return ""
    top = max(weights) or 1.0
    spans = []
    for t, w in zip(tokens, weights):
        a = min(1.0, w / top) * 0.75
        spans.append(
            f'<span style="background:rgba(255,140,0,{a:.2f});padding:2px 4px;'
            f'border-radius:4px;margin:1px;display:inline-block">{html.escape(t)}</span>'
        )
    return " ".join(spans)


@st.cache_data
def load_comparison():
    p = METRIC_DIR / "comparison.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


NICE = {"bilstm_attn": "BiLSTM+Attn", "bilstm_aug": "BiLSTM+Aug", "muril": "MuRIL",
        "xlmr": "XLM-R", "indicbert": "IndicBERTv2"}


st.set_page_config(page_title="Romanized Punjabi Hate Speech", page_icon="🛡️", layout="wide")
st.title("Hate Speech Detection in Romanized Punjabi-English Text")
st.caption("Minor project - BiLSTM+Attention vs MuRIL / XLM-R / IndicBERTv2, "
           "evaluated on authentic YouTube comments.")

tab_demo, tab_results, tab_about = st.tabs(["Live demo", "Results", "About & limitations"])

# ------------------------------------------------------------------ demo
with tab_demo:
    st.warning(
        "**Read this first:** on authentic Punjabi data this model misses most hate "
        "(hate-recall ~12-44%, macro-F1 <= 0.51). It reliably recognises benign text but "
        "often labels clearly hateful text as `not_hate`. This is a research demo, not a "
        "moderation tool."
    )
    available = {k: v for k, v in MODELS.items() if (MODEL_DIR / v).exists()}
    if not available:
        st.error("No model checkpoints found in /models.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c2:
        mname = st.selectbox("Model", list(available))
        thr = st.slider("Hate decision threshold", 0.05, 0.95, 0.50, 0.05,
                        help="Lower it to catch more hate (higher recall, more false alarms).")
        st.caption("Models are biased toward `not_hate`; try lowering the threshold.")
    with c1:
        ex = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
        text = st.text_area("Romanized Punjabi / Punjabi-English text",
                            value="" if ex == "(type your own)" else ex, height=110)

    if st.button("Classify", type="primary") and text.strip():
        cleaned, toks, probs, w = predict(text, available[mname])
        p_hate = probs[1]
        label = "hate" if p_hate >= thr else "not_hate"
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label.upper())
        m2.metric("P(hate)", f"{p_hate:.2f}")
        m3.metric("P(not_hate)", f"{probs[0]:.2f}")
        st.progress(min(max(p_hate, 0.0), 1.0))
        st.markdown("**Words the model attended to** (darker = more weight):")
        st.markdown(highlight(toks, w), unsafe_allow_html=True)
        with st.expander("Preprocessed text"):
            st.code(cleaned)

# ------------------------------------------------------------------ results
with tab_results:
    comp = load_comparison()
    if comp.empty:
        st.info("comparison.csv not found.")
    else:
        comp = comp.assign(model=comp["model"].map(lambda m: NICE.get(m, m)))
        st.subheader("Authentic real test - 1,591 YouTube comments (57 hate)")
        real = comp[comp.eval_on == "rp_real_test"][
            ["model", "train_variant", "macro_f1", "hate_f1", "hate_recall"]
        ].sort_values("macro_f1", ascending=False).round(3)
        st.dataframe(real, hide_index=True, width="stretch")
        st.subheader("In-domain test (Hinglish) - macro-F1")
        ind = comp[comp.eval_on == "indomain_test"].pivot_table(
            index="model", columns="train_variant", values="macro_f1").round(3)
        st.dataframe(ind, width="stretch")

    st.subheader("Robustness to noise (macro-F1 under perturbation)")
    rows = []
    for tag, nm in [("bilstm_attn", "BiLSTM+Attn"), ("muril", "MuRIL"),
                    ("xlmr", "XLM-R"), ("indicbert", "IndicBERTv2"),
                    ("bilstm_aug", "BiLSTM+Aug")]:
        p = METRIC_DIR / f"robustness_{tag}_full.csv"
        if p.exists():
            r = pd.read_csv(p).set_index("perturbation")["macro_f1"].rename(nm)
            rows.append(r)
    if rows:
        st.dataframe(pd.concat(rows, axis=1).round(3), width="stretch")

    st.subheader("Figures")
    figs = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    cols = st.columns(2)
    for i, f in enumerate(figs):
        with cols[i % 2]:
            st.image(str(f), caption=f.stem, width="stretch")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
### Data
- **PRISM Hinglish** (HuggingFace, MIT) - Hindi-English code-mix, used for transfer training.
- **IndicGuard Punjabi** - Gurmukhi safety-prompt text, transliterated to Roman (94% of the
  Romanized Punjabi training volume).
- **Authentic YouTube comments** - collected via the official YouTube Data API v3 (text only,
  handles stripped). Held out of training; used as the real test set.

### Pipeline
`data_loader` -> `preprocess` -> `make_splits` / `build_romanized_punjabi_dataset` ->
`train_bilstm` (local) + `train_transformer` (Colab GPU) -> `robustness_eval` ->
`train_augmented` -> `error_analysis` -> `compare_results` -> `make_report`.

### Limitations
- Only **57 authentic hate examples** exist in the real test set; training data is mostly
  transliterated safety prompts, not social media. This, not model choice, is the bottleneck.
- Transformers (MuRIL, XLM-R, IndicBERTv2) were evaluated in the report but are **not served
  here** (checkpoints are ~700 MB each); this demo serves the BiLSTM models only.
- Not for real moderation decisions.
        """
    )
