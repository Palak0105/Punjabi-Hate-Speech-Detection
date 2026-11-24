"""Robustness perturbations for Romanized Punjabi-English text.

This module is the technical core of the project's novelty (synopsis RQ2/RQ3).
Instead of only reporting clean F1, we apply controlled, realistic noise to the
test set and measure the DROP in performance. The robustness score of a model is:

        robustness = F1(perturbed) / F1(clean)          # closer to 1.0 = better
        degradation = F1(clean) - F1(perturbed)

Each perturbation mimics a real phenomenon in Romanized Punjabi social media:
  * char_elongation : "bura" -> "buraaa"      (emphasis)
  * char_swap       : "veere" -> "veree"      (typos)
  * vowel_drop      : "bhoot" -> "bht"         (fast typing / abbreviation)
  * spelling_variant: dictionary of common Romanized Punjabi variants
  * leet_noise      : random digit/char substitution
  * code_mix_inject : insert common English fillers (yrr, bro, omg)

All functions are seeded and return a NEW string (originals untouched).
"""
from __future__ import annotations

import random
import re

# Common Romanized Punjabi spelling variants (extend during data exploration).
# Maps a canonical-ish form to realistic alternates seen on social media.
SPELLING_VARIANTS = {
    "nahi": ["nai", "nhi", "nahii", "nahin"],
    "tuhada": ["tuhadda", "tuada", "thada"],
    "veere": ["veera", "vere", "vir", "veerey"],
    "bhoot": ["bhot", "bohot", "boht", "bahut"],
    "vadiya": ["vadia", "wadiya", "badiya", "vdiya"],
    "kharab": ["kharaab", "krab", "kharb"],
    "faltu": ["faltoo", "phaltu", "fltu"],
    "paji": ["pajji", "paaji", "pji"],
    "changa": ["chnga", "vadiya", "changa"],
    "bura": ["buraa", "bra", "burra"],
}

ENGLISH_FILLERS = ["yrr", "bro", "omg", "lol", "bhai", "plz", "ok"]
_VOWELS = set("aeiou")


def _rng(seed):
    return random.Random(seed)


def char_elongation(text, *, p=0.3, seed=0):
    r = _rng(seed)
    out = []
    for w in text.split():
        if len(w) > 2 and r.random() < p:
            i = r.randrange(len(w))
            if w[i].isalpha():
                w = w[:i] + w[i] * r.randint(2, 3) + w[i + 1 :]
        out.append(w)
    return " ".join(out)


def char_swap(text, *, p=0.25, seed=0):
    r = _rng(seed)
    out = []
    for w in text.split():
        if len(w) > 3 and r.random() < p:
            i = r.randrange(len(w) - 1)
            lst = list(w)
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            w = "".join(lst)
        out.append(w)
    return " ".join(out)


def vowel_drop(text, *, p=0.3, seed=0):
    r = _rng(seed)
    out = []
    for w in text.split():
        if len(w) > 3 and r.random() < p:
            w = "".join(
                c for j, c in enumerate(w)
                if c.lower() not in _VOWELS or j == 0 or r.random() > 0.6
            )
        out.append(w or "_")
    return " ".join(out)


def spelling_variant(text, *, p=0.5, seed=0):
    r = _rng(seed)
    out = []
    for w in text.split():
        key = w.lower()
        if key in SPELLING_VARIANTS and r.random() < p:
            w = r.choice(SPELLING_VARIANTS[key])
        out.append(w)
    return " ".join(out)


def leet_noise(text, *, p=0.15, seed=0):
    table = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
    r = _rng(seed)
    return "".join(
        table.get(c.lower(), c) if c.lower() in table and r.random() < p else c
        for c in text
    )


def code_mix_inject(text, *, p=0.4, seed=0):
    r = _rng(seed)
    if r.random() < p:
        words = text.split()
        pos = r.randrange(len(words) + 1)
        words.insert(pos, r.choice(ENGLISH_FILLERS))
        return " ".join(words)
    return text


PERTURBATIONS = {
    "char_elongation": char_elongation,
    "char_swap": char_swap,
    "vowel_drop": vowel_drop,
    "spelling_variant": spelling_variant,
    "leet_noise": leet_noise,
    "code_mix_inject": code_mix_inject,
}


def apply_all(text, *, seed=0):
    """Stacked worst-case perturbation (all noise types in sequence)."""
    for i, fn in enumerate(PERTURBATIONS.values()):
        text = fn(text, seed=seed + i)
    return text


def make_perturbed_sets(texts, *, seed=0):
    """Return {perturbation_name: [perturbed_texts]} plus an 'all' variant."""
    out = {}
    for name, fn in PERTURBATIONS.items():
        out[name] = [fn(t, seed=seed) for t in texts]
    out["all"] = [apply_all(t, seed=seed) for t in texts]
    return out


if __name__ == "__main__":
    s = "veere tu bhoot bura banda hai nahi changa"
    print("ORIG:", s)
    for name, fn in PERTURBATIONS.items():
        print(f"{name:16s}:", fn(s, seed=1))
    print(f"{'all':16s}:", apply_all(s, seed=1))
