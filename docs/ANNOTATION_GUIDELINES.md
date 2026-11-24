# Annotation Guidelines — Romanized Punjabi-English Hate Speech

**Version 1.0** · For the minor project *Robust Hate Speech Detection in Romanized
Punjabi-English Code-Mixed Social Media Text*.

These guidelines define how annotators assign labels. Consistency here is what
makes the dataset defensible, so read the whole document before labelling and
return to the decision rules whenever you are unsure.

---

## 1. Task

For each social-media comment, assign a **primary binary label**:

| Label | Value | Meaning |
|-------|-------|---------|
| `not_hate` | 0 | The comment is not hate speech (even if rude or profane). |
| `hate` | 1 | The comment is hate speech (see definition below). |

You will also fill a few **secondary fields** (Section 6) used for analysis only —
they do **not** change the binary label.

---

## 2. Definition of hate speech

> **Hate speech** = any content that **attacks, dehumanises, threatens, incites
> harm against, or promotes hatred toward a person or group *because of a
> protected identity*.**

**Protected identities** (the attack must target one of these):
- Religion / religious practice (e.g. Sikh, Hindu, Muslim, Christian)
- Caste
- Gender or sex
- Sexual orientation / gender identity
- Ethnicity, nationality, region (e.g. Punjabi, Bihari, "outsiders")
- Disability or disease
- Other inherent group identity

A comment is **hate (1)** if it does **any** of these toward a protected group/person:
- Uses slurs or dehumanising language ("animals", "vermin", sub-human terms).
- Calls for or wishes violence, death, expulsion, or harm.
- Promotes discrimination, exclusion, or inherent inferiority/superiority.
- Incites others to attack or harass the group.
- Degrades using identity-based stereotypes with hostile intent.

---

## 3. What is NOT hate speech (label 0)

These are commonly **mislabelled** — be careful:

- **Profanity / insults with no protected-identity target.**
  *"Tu bilkul faltu banda hai"* (you are a useless person) → **0**. Rude, not hate.
- **Criticism of ideas, policies, governments, or public actions.**
  *"Eh policy bakwaas hai"* (this policy is rubbish) → **0**.
- **General anger, venting, or sarcasm** not targeting a protected group → **0**.
- **Reclaimed / in-group usage** with no hostile intent → usually **0** (see 5).
- **Counter-speech** that *quotes* hate to condemn it → **0** (see 5).
- **Offensive jokes** that do not attack a protected identity → **0**.

> Rule of thumb: **Offensive ≠ Hate.** Hate requires a *protected-identity target*
> **and** hostile/dehumanising intent. If either is missing, it is `not_hate`.

---

## 4. Decision procedure (apply in order)

1. **Is a protected identity targeted?** No → `not_hate (0)`.
2. **Is the intent hostile / dehumanising / threatening / exclusionary?**
   No → `not_hate (0)`.
3. Both yes → `hate (1)`.
4. **Unsure after the above?** Use the `uncertain` flag and add a note; do **not**
   guess silently. Uncertain items go to adjudication.

---

## 5. Hard cases & rules

- **Sarcasm / irony:** label by the *actual message*. Sarcastic praise that
  conveys contempt for a protected group → `hate`. Mark `is_sarcasm = 1`.
- **Quoting / reporting:** if the author quotes hate to **condemn** it →
  `not_hate`. If they quote to **endorse/amplify** it → `hate`.
- **Reclaimed slurs:** in-group, non-hostile use → `not_hate`. Same word used to
  attack → `hate`. Judge by target + intent, not the word alone.
- **Implicit hate:** coded language, dog-whistles, "those people should go back"
  → `hate` if the targeted group is clear from context.
- **Mixed comment:** if *any* part is hate speech by the definition → `hate`.
- **Threats against individuals** *because of* their group identity → `hate`.
  Personal threats with no identity basis → `not_hate` (but may flag `uncertain`).
- **Profane slurs toward a protected group** → `hate` even if also "just an insult".

---

## 6. Secondary fields (analysis only — do not affect the binary label)

| Field | Values | Note |
|-------|--------|------|
| `target_type` | none, religion, caste, gender, sexual_orientation, region_ethnicity, disability, political, individual, other | For `not_hate`, use `none` (or `individual`/`political` if insulting but not hateful). |
| `is_sarcasm` | 0 / 1 | Comment relies on sarcasm/irony. |
| `is_code_mixed` | 0 / 1 | Contains both Punjabi and English words. |
| `uncertain` | 0 / 1 | You could not decide confidently → goes to adjudication. |
| `notes` | free text | Reasoning, especially for hard/uncertain cases. |

---

## 7. Romanized Punjabi notes

- **Spelling is not standardised.** The same word appears many ways
  (*nahi / nai / nhi*). Label by meaning, not spelling.
- **Code-mixing is normal.** Punjabi + English in one comment is expected; it does
  not by itself make a comment hateful.
- **Transliteration ambiguity:** if you genuinely cannot read the Romanized
  Punjabi, mark `uncertain` rather than guessing.
- Keep the **original text unchanged** — do not fix spelling in the data.

---

## 8. Example annotations

> Examples are illustrative; the strongest slurs are masked as `[slur]`. Judge
> real comments by the rules above, not by matching these exact strings.

| Text (Romanized Punjabi-English) | Label | target_type | Why |
|---|---|---|---|
| *Eh banda bakwaas kaam karda, bilkul faltu* | 0 | none | Insult, no protected target. |
| *Government di eh scheme ekdum useless hai yaar* | 0 | none | Criticism of policy. |
| *Saare [religion] log terrorist hunde ne, inhe kaddo desh chon* | 1 | religion | Group stereotype + call to expel. |
| *Veere tera kaam vadiya, keep it up* | 0 | none | Positive. |
| *[caste] de log te insaan hi nahi, animals ne* | 1 | caste | Dehumanisation by caste. |
| *Oye tu pagal hai kya, dimaag kharab* | 0 | individual | Personal insult, no protected identity. |
| *Kudiyan nu ghar ch rehna chahida, inha di koi aukaat nahi* | 1 | gender | Gender-based degradation. |
| *Ehna [region] waleyan nu maaro, saade ethe jagah nahi* | 1 | region_ethnicity | Incitement to violence by region. |
| *Someone said "[slur] maro" — this is disgusting, report karo* | 0 | none | Counter-speech condemning hate. |

---

## 9. Workflow

1. Two annotators (**A** and **B**) label the **same** items independently using
   their own sheet — no discussion during labelling.
2. Compute agreement with `src/agreement.py` (Cohen's κ on the binary label).
3. **Adjudicate** every disagreement and every `uncertain = 1` item together (or
   with a third reviewer) to produce the **gold label**.
4. Target: **κ ≥ 0.6** (acceptable), **≥ 0.75** (good). If lower, refine these
   guidelines and re-label a sample.

## 10. Wellbeing

This content can be distressing. Take breaks, label in short sessions, and stop
if needed. The data is handled for research only and is anonymised (no usernames).
