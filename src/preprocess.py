"""Text preprocessing for Romanized Punjabi-English code-mixed social-media text.

Design goals (ties directly to synopsis Section 6.2):
  * Handle social-media noise: URLs, @mentions, #hashtags, emojis, HTML.
  * Normalise spelling noise: character elongation ("bhoottt" -> "bhoot"),
    repeated punctuation ("!!!!" -> "!").
  * Preserve code-mixed content: we DO NOT translate or drop non-English words.
  * Two cleaning levels:
      - "transformer": light clean, keeps casing option off, keeps most tokens
        (subword tokenizers handle noise well).
      - "bilstm": aggressive clean for the word-level neural baseline.

Usage:
    from preprocess import clean_text, Preprocessor
    clean_text("Oye @user tu bhoottt bura hai!!! 😡 http://x.co")
"""
from __future__ import annotations

import html
import re
import unicodedata

try:
    import emoji as _emoji
    _HAS_EMOJI = True
except ImportError:  # keep module importable without the optional dep
    _HAS_EMOJI = False

# --- regex patterns (compiled once) ----------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
# 3+ repeated word characters -> keep 2  (coooool -> cool, bhoottt -> bhoot)
ELONG_RE = re.compile(r"(\w)\1{2,}")
# 2+ repeated punctuation -> keep 1
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")
# keep letters, digits, spaces, and a small set of punctuation
NONTEXT_RE = re.compile(r"[^\w\s!?.,'’]", flags=re.UNICODE)


def _normalise_unicode(text: str) -> str:
    """NFKC fold so visually identical chars map together."""
    return unicodedata.normalize("NFKC", text)


def _handle_emoji(text: str, mode: str) -> str:
    """mode: 'remove' | 'demojize' | 'keep'."""
    if not _HAS_EMOJI or mode == "keep":
        return text
    if mode == "demojize":
        # :angry_face: -> angry_face  (emojis can carry sentiment signal)
        return re.sub(r":([a-z_]+):", r" \1 ", _emoji.demojize(text))
    return _emoji.replace_emoji(text, replace=" ")


def clean_text(
    text: str,
    *,
    level: str = "transformer",
    lowercase: bool | None = None,
    emoji_mode: str = "demojize",
    keep_hashtag_word: bool = True,
) -> str:
    """Clean a single social-media string.

    Parameters
    ----------
    level : 'transformer' (light) or 'bilstm' (aggressive).
    lowercase : force case; default True for bilstm, False for transformer.
    emoji_mode : 'demojize' (default), 'remove', or 'keep'.
    keep_hashtag_word : '#punjab' -> 'punjab' when True, else dropped.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if lowercase is None:
        lowercase = level == "bilstm"

    t = _normalise_unicode(text)
    t = html.unescape(t)
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)                       # drop handles (noise, PII)
    t = HASHTAG_RE.sub(r" \1 " if keep_hashtag_word else " ", t)
    t = _handle_emoji(t, emoji_mode)

    t = ELONG_RE.sub(r"\1\1", t)                     # bhoottt -> bhoott
    t = REPEAT_PUNCT_RE.sub(r"\1", t)                # !!!! -> !

    if level == "bilstm":
        t = NONTEXT_RE.sub(" ", t)                   # strip residual symbols

    if lowercase:
        t = t.lower()

    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


class Preprocessor:
    """Vectorised wrapper so you can `df['clean_text'] = pp(df['text'])`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, series):
        return series.astype(str).map(lambda x: clean_text(x, **self.kwargs))


if __name__ == "__main__":
    samples = [
        "Oye @randomUser tuhaaadaa dimaag kharab hai!!! 😡😡 http://spam.co #Punjab",
        "bhoottt vadiya kaam kita veere 👍",
        "<p>Ehh banda bilkul faltu hai yrr</p>",
    ]
    for s in samples:
        print("RAW  :", s)
        print("TFMR :", clean_text(s, level="transformer"))
        print("LSTM :", clean_text(s, level="bilstm"))
        print("-" * 60)
