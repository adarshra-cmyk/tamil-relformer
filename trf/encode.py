"""Typed entity markers and model inputs (plan section 6, Steps A and C).

Baseline (E0) input:
    <SUBJ-PER> சாம் </SUBJ-PER> <OBJ-LOC> யாழ்ப்பாணத்தில் </OBJ-LOC> வாழ்ந்தார்.

Morphology variants only ADD tokens; the sentence and typed markers stay the
same as E0, so differences between runs come from the added information:
    lemma_in_text (E1):  <OBJ-LOC> யாழ்ப்பாணத்தில் <LEM> யாழ்ப்பாணம் </OBJ-LOC>
    case_in_text  (E2a): <OBJ-LOC> <CASE-Loc> யாழ்ப்பாணத்தில் </OBJ-LOC>
    case_embedding (E2b): no text change; case ids go to a learned embedding
                          that is concatenated with the entity-pair vector.
The entity representation is the encoder state at the start marker
(Zhou & Chen 2022, typed entity marker).
"""
from __future__ import annotations

from dataclasses import dataclass

from .morph import CASE_VOCAB
from .schema import ENTITY_TYPES, TYPE_ABBR, Example

MARKER_TOKENS = (
    [f"<{r}-{TYPE_ABBR[t]}>" for r in ("SUBJ", "OBJ") for t in sorted(ENTITY_TYPES)]
    + [f"</{r}-{TYPE_ABBR[t]}>" for r in ("SUBJ", "OBJ") for t in sorted(ENTITY_TYPES)]
    + ["<LEM>"] + [f"<CASE-{c}>" for c in CASE_VOCAB]
)


@dataclass
class MorphConfig:
    lemma_in_text: bool = False
    case_in_text: bool = False
    case_embedding: bool = False


# Descriptions used by marker_init="informed": each added token's embedding starts as the
# mean input embedding of its description, instead of the near-identical rows that
# resize_token_embeddings gives every new token (all markers would start as one vector).
_TYPE_WORD = {"PERSON": "person", "LOCATION": "location", "ORGANIZATION": "organization",
              "DATE": "date", "MISC": "other"}
_CASE_TEXT = {"Nom": "nominative", "Acc": "ஐ", "Dat": "க்கு", "Loc": "இல்", "Ins": "ஆல்",
              "Com": "உடன்", "Abl": "இலிருந்து", "Gen": "இன்", "Ben": "க்காக", "Voc": "vocative",
              "<NONE>": "none", "<MISSING>": "missing", "OTHER": "adjective"}


def marker_descriptions() -> dict[str, str]:
    d = {}
    for r, word in (("SUBJ", "subject"), ("OBJ", "object")):
        for t in ENTITY_TYPES:
            d[f"<{r}-{TYPE_ABBR[t]}>"] = f"{word} {_TYPE_WORD[t]}"
            d[f"</{r}-{TYPE_ABBR[t]}>"] = f"end {word} {_TYPE_WORD[t]}"
    d["<LEM>"] = "lemma"
    for c in CASE_VOCAB:
        d[f"<CASE-{c}>"] = _CASE_TEXT[c]
    assert set(d) == set(MARKER_TOKENS)
    return d


def add_marker_tokens(tokenizer) -> int:
    return tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})


def morph_for(ex: Example, role: str) -> dict:
    """Verified morphology beats predicted; returns {} when neither exists."""
    eid = ex.subject if role == "subj" else ex.object
    return ex.meta.get("morph", {}).get(eid) or ex.meta.get("morph_pred", {}).get(role) or {}


def marked_text(ex: Example, mc: MorphConfig) -> str:
    spans = [("SUBJ", ex.subj, morph_for(ex, "subj")), ("OBJ", ex.obj, morph_for(ex, "obj"))]
    spans.sort(key=lambda s: s[1].start)
    if spans[0][1].end > spans[1][1].start:
        raise ValueError(f"overlapping entities in {ex.source_id}: {ex.sentence}")
    out, pos = [], 0
    for role, ent, m in spans:
        tag = TYPE_ABBR[ent.type]
        out.append(ex.sentence[pos:ent.start])
        out.append(f" <{role}-{tag}> ")
        if mc.case_in_text:
            out.append(f"<CASE-{m.get('case', '<MISSING>')}> ")
        out.append(ent.text)
        if mc.lemma_in_text and m.get("lemma"):
            out.append(f" <LEM> {m['lemma']}")
        out.append(f" </{role}-{tag}> ")
        pos = ent.end
    out.append(ex.sentence[pos:])
    return " ".join("".join(out).split())


def case_ids(ex: Example) -> tuple[int, int]:
    def cid(role):
        return CASE_VOCAB.index(morph_for(ex, role).get("case", "<MISSING>"))
    return cid("subj"), cid("obj")


def suffix_positions(ex: Example, tokenizer, ids: list[int], offsets: list[tuple[int, int]], text: str) -> list[int]:
    """Token positions holding the inflectional suffix of each entity's surface form.

    The stem is the longest common prefix of the surface form and its lemma; tokens
    of the surface form that end after the stem are suffix tokens (a token that
    straddles the boundary counts as suffix). Used by morphology-guided masking (E3).
    """
    out = []
    for role, ent in (("SUBJ", ex.subj), ("OBJ", ex.obj)):
        lemma = morph_for(ex, role.lower()).get("lemma") or ""
        k = 0
        while k < min(len(lemma), len(ent.text)) and lemma[k] == ent.text[k]:
            k += 1
        if not lemma or k >= len(ent.text):
            continue                                  # no suffix (or no lemma to tell)
        marker = tokenizer.convert_tokens_to_ids(f"<{role}-{TYPE_ABBR[ent.type]}>")
        if marker not in ids:
            continue
        # the surface form is the first occurrence of ent.text after the start marker
        m_end = offsets[ids.index(marker)][1]
        start = text.find(ent.text, m_end)
        if start < 0:
            continue
        cut, end = start + k, start + len(ent.text)
        out += [i for i, (a, b) in enumerate(offsets) if a < end and b > cut and b > a]
    return out


def encode(ex: Example, tokenizer, mc: MorphConfig, label2id: dict, max_len: int = 256,
           with_suffix: bool = False) -> dict:
    text = marked_text(ex, mc)
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=with_suffix)
    ids = enc["input_ids"]
    s_tok = tokenizer.convert_tokens_to_ids(f"<SUBJ-{TYPE_ABBR[ex.subj.type]}>")
    o_tok = tokenizer.convert_tokens_to_ids(f"<OBJ-{TYPE_ABBR[ex.obj.type]}>")
    if s_tok not in ids or o_tok not in ids:
        raise ValueError(f"entity marker truncated (max_len={max_len}) in {ex.source_id}")
    sc, oc = case_ids(ex)
    out = {
        "input_ids": ids,
        "subj_pos": ids.index(s_tok),
        "obj_pos": ids.index(o_tok),
        "subj_case": sc,
        "obj_case": oc,
        "label": label2id[ex.label],
    }
    if with_suffix:
        out["suffix_pos"] = suffix_positions(ex, tokenizer, ids, enc["offset_mapping"], text)
    return out
