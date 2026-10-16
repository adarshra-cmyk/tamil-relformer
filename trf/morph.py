"""Morphological features for entity spans (plan section 6, Step B).

An analyzer turns a sentence into tokens with character offsets, lemma, UPOS
and Case. `entity_features` then picks the feature record for one entity span.
Tamil is head-final, so the case of a multi-token entity is read from its last
token.

Every record carries a `status` so the model and the analysis can tell apart
analyzer-predicted, manually verified and missing values (plan: "Record whether
a feature is analyzer-predicted, manually verified, missing or ambiguous").
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .schema import Entity, Example

# UD Case values we keep as distinct classes. Anything else maps to OTHER.
CASES = ["Nom", "Acc", "Dat", "Loc", "Ins", "Com", "Abl", "Gen", "Ben", "Voc"]
CASE_VOCAB = ["<NONE>", "<MISSING>", "OTHER"] + CASES

# Stanza's Tamil model splits some case forms into host + postposition word
# (e.g. கொழும்பிலிருந்து -> கொழும்ப் + இலிருந்து). When the postposition sits
# inside the same surface token, it determines the case of the token.
POSTPOSITION_CASE = {
    "இலிருந்து": "Abl", "லிருந்து": "Abl", "இருந்து": "Abl",
    "உடன்": "Com", "ஓடு": "Com", "ஆல்": "Ins",
}


@dataclass
class Token:
    text: str
    start: int
    end: int
    lemma: str
    upos: str
    case: str | None
    ambiguous: bool = False


@dataclass
class MorphFeatures:
    lemma: str | None
    case: str          # one of CASE_VOCAB
    upos: str | None
    status: str        # predicted | verified | missing

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_feats(feats: str | None) -> dict:
    if not feats:
        return {}
    return dict(kv.split("=", 1) for kv in feats.split("|") if "=" in kv)


class StanzaAnalyzer:
    """Stanza `ta` pipeline (UD Tamil-TTB). Caches results on disk by sentence hash."""

    name = "stanza-ta"

    def __init__(self, cache_dir: str | Path = "data/processed/morph_cache", use_gpu: bool = False):
        import stanza
        stanza.download("ta", processors="tokenize,mwt,pos,lemma", verbose=False)
        self.nlp = stanza.Pipeline("ta", processors="tokenize,mwt,pos,lemma",
                                   use_gpu=use_gpu, verbose=False)
        self.cache_dir = Path(cache_dir) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, sentence: str) -> list[Token]:
        key = hashlib.sha1(sentence.encode("utf-8")).hexdigest()
        cpath = self.cache_dir / f"{key}.json"
        if cpath.exists():
            return [Token(**t) for t in json.loads(cpath.read_text(encoding="utf-8"))]
        doc = self.nlp(sentence)
        toks = []
        for sent in doc.sentences:
            for tok in sent.tokens:
                words = tok.words
                host = next((w for w in words if w.upos != "ADP"), words[0])
                case = _parse_feats(host.feats).get("Case")
                for w in words[1:]:
                    if w.upos == "ADP" and w.text in POSTPOSITION_CASE:
                        case = POSTPOSITION_CASE[w.text]
                toks.append(Token(tok.text, tok.start_char, tok.end_char,
                                  host.lemma or host.text, host.upos, case))
        cpath.write_text(json.dumps([asdict(t) for t in toks], ensure_ascii=False), encoding="utf-8")
        return toks


class ThamizhiMorphAnalyzer:
    """ThamizhiMorph foma FSTs (Sarveswaran et al. 2021) via `flookup`.

    Whitespace tokenization; each word goes through the noun and pronoun
    lexicon FSTs, then the noun guesser. Lexicon FSTs only know listed stems,
    so most proper names reach the guesser, whose lemmas can be truncated.
    When analyses disagree on case, the first lexicon analysis is kept and the
    token is flagged ambiguous.
    """

    name = "thamizhimorph"
    TAG2CASE = {"nom": "Nom", "acc": "Acc", "dat": "Dat", "loc": "Loc", "abl": "Abl",
                "gen": "Gen", "inst": "Ins", "soc": "Com", "ben": "Ben", "voc": "Voc"}

    def __init__(self, fst_dir: str | Path = "data/raw/thamizhi-morph/FST-Models"):
        self.fst_dir = Path(fst_dir)
        self.fsts = [("lex", self.fst_dir / "noun.fst"), ("lex", self.fst_dir / "pronoun.fst"),
                     ("guess", self.fst_dir / "noun-guess.fst")]
        self._memo: dict[str, list[tuple[str, str]]] = {}

    def _lookup_all(self, words: list[str]) -> None:
        import subprocess
        todo = [w for w in dict.fromkeys(words) if w not in self._memo]
        for w in todo:
            self._memo[w] = []
        for kind, fst in self.fsts:
            pending = [w for w in todo if not self._memo[w]]
            if not pending:
                break
            out = subprocess.run(["flookup", str(fst)], input="\n".join(pending) + "\n",
                                 capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                if "\t" not in line:
                    continue
                w, ana = line.split("\t", 1)
                if ana != "+?" and w in self._memo:
                    self._memo[w].append((kind, ana))

    def analyze(self, sentence: str) -> list[Token]:
        spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[^\s.,;:!?\"'()\[\]]+", sentence)]
        self._lookup_all([w for w, _, _ in spans])
        toks = []
        for w, s, e in spans:
            anas = self._memo.get(w, [])
            if not anas:
                toks.append(Token(w, s, e, w, None, None))
                continue
            parsed = []
            for kind, a in anas:
                parts = a.split("+")
                case = next((self.TAG2CASE[p] for p in parts[1:] if p in self.TAG2CASE), None)
                pos = "PRON" if "pronoun" in parts else "NOUN"
                parsed.append((parts[0], pos, case))
            lemma, pos, case = parsed[0]
            ambiguous = len({c for _, _, c in parsed}) > 1
            toks.append(Token(w, s, e, lemma, pos, case, ambiguous))
        return toks


def _norm_case(case: str | None) -> str:
    if case is None:
        return "<NONE>"
    return case if case in CASES else "OTHER"


def entity_features(tokens: list[Token], ent: Entity) -> MorphFeatures:
    inside = [t for t in tokens if t.start < ent.end and t.end > ent.start]
    if not inside:
        return MorphFeatures(None, "<MISSING>", None, "missing")
    head = inside[-1]
    lemma_parts = [t.text for t in inside[:-1]] + [head.lemma]
    if head.case is None and head.upos is None:
        return MorphFeatures(None, "<MISSING>", None, "missing")
    status = "ambiguous" if head.ambiguous else "predicted"
    return MorphFeatures(" ".join(lemma_parts), _norm_case(head.case), head.upos, status)


def example_features(ex: Example, analyzer) -> dict[str, MorphFeatures]:
    """Features for subject and object. Verified morphology in
    ex.meta["morph"][entity_id] (keys lemma/case/upos) overrides the analyzer."""
    verified = ex.meta.get("morph", {})
    out = {}
    tokens = None
    for role, ent in (("subj", ex.subj), ("obj", ex.obj)):
        if ent.id in verified:
            v = verified[ent.id]
            out[role] = MorphFeatures(v.get("lemma"), _norm_case(v.get("case")), v.get("upos"), "verified")
            continue
        if tokens is None:
            tokens = analyzer.analyze(ex.sentence)
        out[role] = entity_features(tokens, ent)
    return out
