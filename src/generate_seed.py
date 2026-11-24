"""Author a Romanized Punjabi-English SEED dataset for development.

WHY THIS EXISTS
---------------
No public labelled Romanized Punjabi-English hate-speech dataset exists (verified
Nov 2026). Real collection needs a YouTube Data API key + human annotation
(pipeline: collect_youtube.py -> make_annotation_sheet.py -> agreement.py).
Until that gold set exists, this module provides a *development seed*: realistic,
hand-authored examples so the model pipeline (Stages 2-6) can be built and tested.

HONESTY / LIMITATIONS  (state these in the report)
  * This is author-generated, NOT collected social-media data.
  * Use it as a DEV/smoke-test set and for augmentation — not as the final
    evaluation set. The real gold set (punjabi_gold.csv) replaces it for results.
  * Every row is labelled source='seed_synth' so it never silently mixes with
    real data.

The hateful examples are realistic but measured; the strongest slurs are written
as `[SLUR]` placeholders. Content is for building a *detector* (defensive use).

Output: data/processed/punjabi_seed.csv  (id, text, label, target_type,
        is_code_mixed, source)

Run:
    python src/generate_seed.py                 # base + light augmentation
    python src/generate_seed.py --no-augment    # base examples only
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# ---------------------------------------------------------------------------
# BASE EXAMPLES  (text, label, target_type)
# label: 0 = not_hate, 1 = hate   (see docs/ANNOTATION_GUIDELINES.md)
# ---------------------------------------------------------------------------
NOT_HATE = [
    # positive / supportive
    ("Veere tera kaam bohot vadiya hai, keep it up", "none"),
    ("Bahut sohna video banaya yaar, mazaa aa gaya", "none"),
    ("Changa laga eh content, thanks for sharing paji", "none"),
    ("Proud of you munde, tusi great kaam kita", "none"),
    ("Eh gaana sunke dil khush ho gaya, superb", "none"),
    ("Tuhada message bilkul sahi hai, fully agree", "none"),
    ("Waah ji waah, kya baat hai, lajawaab", "none"),
    ("God bless you veere, keep spreading positivity", "none"),
    # neutral / questions / everyday
    ("Yaar eh video kithe shoot kita, location dasso", "none"),
    ("Menu samajh nahi aayi, koi explain karo please", "none"),
    ("Kal match kis time start hona, koi pata", "none"),
    ("Eh recipe try karni hai, ingredients dasso", "none"),
    ("Punjab da mausam ajkal bohot vadiya chal reha", "none"),
    ("Koi dasso eh phone da price kinna hai", "none"),
    # criticism of ideas/policy/actions (NOT hate)
    ("Eh government di policy bilkul faltu hai, kuch nahi kita", "political"),
    ("Is party da kaam bakwaas, vote waste ho gaya", "political"),
    ("Movie boring si, paise waste ho gaye, story weak", "none"),
    ("Service bohot kharab hai is company di, refund chahida", "none"),
    ("Traffic system theek karo, roz jam lagda", "none"),
    ("Cricket team ne bura khela aj, selection galat", "none"),
    # personal insult WITHOUT protected identity (offensive, not hate)
    ("Tu pagal hai kya, dimaag kharab ho gaya tera", "individual"),
    ("Oye faltu banda, tenu kuch nahi aunda", "individual"),
    ("Bakwaas na kar, tu bilkul useless hai", "individual"),
    ("Tera dimaag ghaas charan gaya ki", "individual"),
    ("Chup kar be, tenu akal nahi hai bilkul", "individual"),
    # counter-speech / condemning hate
    ("Kise ne hate comment kita si, report karo aisa content", "none"),
    ("Aisi nafrat faila na karo yaar, sab barabar ne", "none"),
    ("Eh galat hai, kise di religion da mazaak nahi udana chahida", "none"),
    # benign sarcasm
    ("Waah bhai waah, bijli roz jaandi, kamaal service", "political"),
]

HATE = [
    # religion
    ("Saare [COMMUNITY] log terrorist hunde ne, inhe desh chon kaddo", "religion"),
    ("Eh [COMMUNITY] waleyan nu jeena da haq nahi, maaro inhe", "religion"),
    ("[COMMUNITY] di aukaat hi nahi, gandagi ne eh log", "religion"),
    ("Inha [COMMUNITY] nu ethe nahi rehna chahida, bhejo wapas", "religion"),
    ("[COMMUNITY] log sab chor te dhokebaaz hunde ne pakka", "religion"),
    # caste
    ("[CASTE] de log te insaan hi nahi, animals warga behaviour", "caste"),
    ("Neech [CASTE] nu saade naal baithan da haq nahi", "caste"),
    ("[CASTE] waleyan di koi izzat nahi, ganda khoon", "caste"),
    ("Aise [CASTE] log saade pind chon bahar hone chahide", "caste"),
    # gender
    ("Kudiyan di koi aukaat nahi, ghar ch band rakho inhe", "gender"),
    ("Aurtan nu dimaag nahi hunda, bas kitchen layi bani ne", "gender"),
    ("Eh kudi bilkul [SLUR] hai, aisiyan nu ghar chon kaddo", "gender"),
    ("Munde hi kaam de ne, kudiyan te bojh hunde ne", "gender"),
    # region / ethnicity
    ("Ehna bahar waleyan nu maaro, saade sheher ch jagah nahi", "region_ethnicity"),
    ("[REGION] de log sab ganday te chor, bhejo wapas pind", "region_ethnicity"),
    ("Inha pravasiyan nu kaddo, saadi naukriyan kha gaye", "region_ethnicity"),
    ("[REGION] waleyan di nasal hi kharab hai, insaan nahi", "region_ethnicity"),
    # sexual orientation
    ("Aise [SLUR] log bimari ne, inha nu theek karo ya maaro", "sexual_orientation"),
    ("Eh log samaj te daag ne, inha nu jeena nahi dena chahida", "sexual_orientation"),
    # disability
    ("Aise apahaj logan nu te maar dena chahida, bojh ne samaj te", "disability"),
    ("Pagal te mandbudhi logan di koi value nahi, hatao inhe", "disability"),
    # dehumanising / incitement (religion/region mix)
    ("Inha [COMMUNITY] nu zinda saad do, koi haq nahi jeena da", "religion"),
    ("Saare [REGION] waleyan nu ghar chon kadd ke maaro", "region_ethnicity"),
    ("[CASTE] di nasal mita deni chahidi, ganda khoon hai eh", "caste"),
    ("Kudiyan nu padhai da haq nahi, inha di jagah ghar hai bas", "gender"),
]

# code-mixed variants that lean more on English (realistic social-media style)
NOT_HATE_MIX = [
    ("Honestly yaar this video is amazing, bohot mehnat kiti", "none"),
    ("Guys please is topic te fight mat karo, respect everyone", "none"),
    ("This policy is useless, government ne kuch deliver nahi kita", "political"),
    ("Bro tu overreact kar raha, chill maar thoda", "individual"),
    ("Such a peaceful place, Punjab di visit must karni chahidi", "none"),
]
HATE_MIX = [
    ("These [COMMUNITY] people should be banned, saade desh chon kaddo", "religion"),
    ("Girls are worthless yaar, inha nu rights deni hi nahi chahidi", "gender"),
    ("All [REGION] migrants are criminals, throw them out saade sheher chon", "region_ethnicity"),
]


def _is_code_mixed(text: str) -> int:
    eng = {"the", "is", "are", "this", "you", "please", "should", "video",
           "people", "all", "guys", "bro", "girls", "keep", "up", "thanks",
           "god", "bless", "proud", "service", "team", "topic", "respect"}
    words = set(w.lower().strip(".,!?") for w in text.split())
    has_eng = len(words & eng) > 0
    # crude: assume Punjabi markers present in all our rows -> code-mix if english too
    return int(has_eng)


def _row_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def build_base() -> pd.DataFrame:
    rows = []
    for text, tgt in NOT_HATE + NOT_HATE_MIX:
        rows.append((text, 0, tgt))
    for text, tgt in HATE + HATE_MIX:
        rows.append((text, 1, tgt))
    df = pd.DataFrame(rows, columns=["text", "label", "target_type"])
    df["is_code_mixed"] = df["text"].map(_is_code_mixed)
    df["source"] = "seed_synth"
    return df


def augment(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Light spelling-variation augmentation so the seed has surface diversity.

    Uses perturb.spelling_variant + char_elongation (label-preserving) to mimic
    Romanized spelling noise. Produces 1 variant per row.
    """
    from perturb import spelling_variant, char_elongation

    aug = []
    for i, r in df.reset_index(drop=True).iterrows():
        v = spelling_variant(r["text"], p=0.6, seed=seed + i)
        v = char_elongation(v, p=0.25, seed=seed + i)
        if v != r["text"]:
            aug.append((v, r["label"], r["target_type"], r["is_code_mixed"], "seed_synth_aug"))
    adf = pd.DataFrame(aug, columns=["text", "label", "target_type", "is_code_mixed", "source"])
    return adf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    base = build_base()
    parts = [base]
    if not args.no_augment:
        parts.append(augment(base, args.seed))
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].str.strip()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.insert(0, "id", df["text"].map(_row_id))

    out = C.PROC_DIR / "punjabi_seed.csv"
    df.to_csv(out, index=False)

    print(f"saved: {out}")
    print(f"rows : {len(df)}  (base={len(base)}, augmented={len(df)-len(base)})")
    print("\nlabel distribution:")
    print(df["label"].map(C.ID2LABEL).value_counts().to_string())
    print("\ntarget_type (hate rows):")
    print(df[df.label == 1]["target_type"].value_counts().to_string())
    print(f"\ncode-mixed rows: {int(df['is_code_mixed'].sum())} / {len(df)}")
    print("\n[!] seed_synth = author-generated DEV data. Replace with punjabi_gold.csv "
          "(real, human-annotated) for final results.")


if __name__ == "__main__":
    main()
