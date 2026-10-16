"""Distant-supervision candidates from Tamil Wikipedia + Wikidata.

For each subject with a Tamil Wikipedia article, split its article into
sentences and keep sentences where a subject mention and an object mention
both occur:
  * positive candidate: the object is linked to the subject by a Wikidata
    fact; label = that relation (all matching relations kept in meta.ds_labels)
  * NO_RELATION candidate: the object is a known entity of a type that fits
    some relation for this subject, but no fact links the two

Mentions are matched on inflected surface forms: யாழ்ப்பாணம் matches
யாழ்ப்பாணத்தில், கொழும்பு matches கொழும்பிலிருந்து, குமார் matches
குமாருக்கு. The entity span is the full inflected word, as in the plan's
example. These are CANDIDATES for annotation: distant supervision is noisy
(the sentence may not express the relation), and every example needs a
human label before it is used for evaluation.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trf.schema import NO_RELATION

PULLI = "்"
VOWEL_SIGNS = set("ாிீுூெேைொோௌ")
# Allowed material after a name or its stem: optional glide (ய/வ after a
# vowel-final name), optional case ending, then clitics or sandhi consonants.
# Anything else (a longer word that happens to share the prefix, e.g.
# பெரு -> பெரும்பாலான) is rejected.
CASE_ENDINGS = ["ிலிருந்து", "ினுடைய", "ிற்கு", "ினால்", "ுக்கு", "ுடைய", "ுடன்", "ிடம்",
                "ினது", "ின்", "ில்", "ால்", "ோடு", "ைய", "ை", "ில", "ின", "ுக்க"]
CLITICS = ["ுள்ள", "ேயே", "ும்", "ான", "ாக", "ே", "ச்", "க்", "ப்", "த்"]
_alt = lambda xs: "|".join(sorted(map(re.escape, xs), key=len, reverse=True))
SUFFIX_RE = re.compile(rf"^(?:[யவ])?(?:{_alt(CASE_ENDINGS)})?(?:{_alt(CLITICS)})*$")
MAX_SUFFIX = 14
WORD_RE = re.compile(r"[^\s.,;:!?\"'()\[\]{}<>“”‘’–—-]+")


def strip_title(t: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", t).strip()


def stems(word: str) -> list[str]:
    """Prefixes that inflected forms of `word` start with."""
    out = [word]
    if word.endswith("ம" + PULLI) and len(word) > 3:          # -am nouns: யாழ்ப்பாணம் -> யாழ்ப்பாணத்தில்
        out.append(word[:-2] + "த்த")
    if word[-1] in VOWEL_SIGNS and word[-1] in "ு" and len(word) > 3:  # final -u: கொழும்பு -> கொழும்ப
        out.append(word[:-1])
    if word.endswith(PULLI) and len(word) > 3:                 # final consonant: குமார் -> குமார
        out.append(word[:-1])
    return out


class Gazetteer:
    """Maps names to entities; matches multi-word names whose last word may be inflected."""

    def __init__(self, min_inflect: int = 0):
        """Single-word names shorter than `min_inflect` characters only match exactly
        (கோய would otherwise match கோயிலாகும் 'is a temple')."""
        self.by_first: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
        self.min_inflect = min_inflect

    def add(self, name: str, key: str) -> None:
        words = WORD_RE.findall(strip_title(name))
        if not words or len("".join(words)) < 3:
            return
        self.by_first[words[0][:2]].append((words, key))

    def find(self, sent_words: list[tuple[str, int, int]], sentence: str | None = None) -> list[tuple[int, int, str, str]]:
        """Returns (char_start, char_end, key, match_type) for every match.
        With `sentence`, words of a multi-word name must be separated by whitespace only."""
        hits = []
        for i, (w, s, _) in enumerate(sent_words):
            for words, key in self.by_first.get(w[:2], ()):
                n = len(words)
                if i + n > len(sent_words):
                    continue
                if any(sent_words[i + k][0] != words[k] for k in range(n - 1)):
                    continue
                if sentence is not None and any(
                        sentence[sent_words[i + k][2]:sent_words[i + k + 1][1]].strip()
                        for k in range(n - 1)):
                    continue
                last = sent_words[i + n - 1][0]
                if n == 1 and len(last) != len(words[0]) and len(words[0]) < self.min_inflect:
                    continue
                for st in (stems(words[-1]) if len(words[-1]) >= 4 else [words[-1]]):
                    if last.startswith(st):
                        suf = last[len(st):]
                        if len(suf) <= MAX_SUFFIX and SUFFIX_RE.match(suf):
                            hits.append((s, sent_words[i + n - 1][2], key,
                                         "exact" if last == words[-1] else "inflected"))
                            break
        return hits


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", para.strip())
        buf = ""
        for p in parts:
            buf = f"{buf} {p}".strip() if buf else p
            last = WORD_RE.findall(buf)[-1:] or [""]
            if buf.endswith(".") and len(last[0]) <= 2:         # initials like "வ. உ." are not sentence ends
                continue
            out.append(buf); buf = ""
        if buf:
            out.append(buf)
    return out


# Words that start a longer institution name after a place name:
# தில்லி உயர்நீதிமன்றம், மஸ்கத் பன்னாட்டு வானூர்தி நிலையம், இந்திய தேசிய காங்கிரஸ்.
# மாவட்டம்/நகரம் are not here: a bare place name before them is a valid span (guide v0.1).
INSTITUTION_HEADS = ("உயர்நீதிமன்ற", "உச்சநீதிமன்ற", "நீதிமன்ற", "பல்கலைக்கழக", "கல்லூரி",
                     "வானூர்தி", "விமான", "பன்னாட்டு", "தொடருந்து", "சட்டமன்ற", "மக்களவை",
                     "கட்சி", "காங்கிரஸ", "அணி", "சங்க", "நிறுவன", "கழக", "பள்ளி", "தேசிய",
                     "ஆட்சிப்", "வங்கி", "மருத்துவமனை", "அருங்காட்சியக", "துறைமுக", "கோயில்")
TITLE_WORDS = {"சர்", "திரு", "திருமதி", "செல்வி", "டாக்டர்", "முனைவர்", "பேராசிரியர்", "மாஸ்டர்",
               "ஸ்ரீ", "சுவாமி", "பகவான்", "புனித", "கடவுள்", "மன்னர்", "மன்னன்", "அரசர்", "அரசன்",
               "அரசி", "ராணி", "இராணி", "பேரரசர்", "இளவரசர்", "இளவரசி", "சுல்தான்", "நடிகர்", "நடிகை",
               "இயக்குநர்", "கவிஞர்", "தம்பி", "அண்ணன்", "தங்கை", "அக்கா", "மகன்", "மகள்", "மனைவி",
               "கணவர்", "தந்தை", "தாய்", "தாயார்", "தந்தையார்", "தலைவர்", "அமைச்சர்", "நீதிபதி",
               "கேப்டன்", "ஜெனரல்", "மேஜர்", "கர்னல்", "டத்தோ", "துன்", "சேக்", "ஷேக்", "இமாம்", "தளபதி", "தாக்கூர்", "ராஜா", "இராஜா", "மகாராஜா", "பண்டிட்"}
INITIALS_RE = re.compile(r"((?:[\u0B80-\u0BFF]{1,3}\.\s?)+)$")


def extend_initials(sent: str, start: int) -> int:
    """Move a person span start left over initials directly before it (சு. சமுத்திரம்)."""
    m = INITIALS_RE.search(sent[:start])
    return m.start(1) if m else start


def name_pairs(names) -> set[tuple[str, str]]:
    """(word, next word) pairs inside a person's names."""
    out = set()
    for n in names:
        ws = WORD_RE.findall(n)
        out.update(zip(ws, ws[1:]))
    return out


def extend_own(sent: str, start: int, pairs: set[tuple[str, str]]) -> int:
    """Move a person span start left over words that come right before it in the
    person's own name (முஹம்மது [யூனூஸ்] -> [முஹம்மது யூனூஸ்]). Titles stay out."""
    while True:
        m = re.search(r"(\S+)\s+$", sent[:start])
        first = WORD_RE.match(sent, start)
        if not m or not first or m.group(1) in TITLE_WORDS or (m.group(1), first.group()) not in pairs:
            return start
        start = m.start(1)


def next_word(words, end):
    return next((w for w, s, _ in words if s >= end), "")


def prev_word(words, start):
    prev = [w for w, _, e in words if e <= start]
    return prev[-1] if prev else ""


def drop_nested(hits):
    """Remove hits strictly inside a longer hit (மாட்ரிட் inside ரியல் மாட்ரிட்)."""
    return [h for h in hits
            if not any(o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/wikidata_facts.jsonl")
    ap.add_argument("--relations", default="configs/relations.json")
    ap.add_argument("--wiki", nargs="+", default=sorted(map(str, Path("data/raw/tawiki").glob("*.parquet"))))
    ap.add_argument("--out", default="data/processed/ds_candidates.jsonl")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--neg-per-article", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--types", default="data/processed/entity_types.json",
                    help="Wikidata-class entity types; slot-derived types are used if missing")
    ap.add_argument("--classes", default="data/processed/entity_classes.json",
                    help="P31 classes per entity; entities with none fall back to the slot type")
    ap.add_argument("--alias-stoplist", default="configs/alias_stoplist.txt",
                    help="single-word aliases that are common words or point to the wrong entity")
    ap.add_argument("--legacy", action="store_true", help="v0 matcher (no pilot fixes), for comparison")
    ap.add_argument("--skip-fix", nargs="*", default=[],
                    choices=["alias_stop", "short_inflect", "surname", "institution", "nested", "initials", "type_check"],
                    help="turn off single pilot fixes, to see what each one drops")
    ap.add_argument("--only-docs", help="file with one article title per line; skip all other articles")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    only_docs = set(Path(a.only_docs).read_text(encoding="utf-8").split("\n")) - {""} if a.only_docs else None

    rels = {r["label"]: r for r in json.loads(Path(a.relations).read_text(encoding="utf-8"))["relations"]}
    facts = [json.loads(l) for l in open(a.facts, encoding="utf-8")]

    # entity info and fact index
    ent_type: dict[str, str] = {}
    ent_names: dict[str, set[str]] = defaultdict(set)
    subj_facts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    subj_title: dict[str, str] = {}
    for f in facts:
        r = rels[f["rel"]]
        ent_type.setdefault(f["s"], r["subj"]); ent_type.setdefault(f["o"], r["obj"])
        ent_names[f["s"]].update({f["s_label"], strip_title(f["s_title"])})
        ent_names[f["o"]].update({f["o_label"], *f["o_aliases"]})
        subj_facts[f["s"]][f["o"]].add(f["rel"])
        subj_title[f["s_title"]] = f["s"]

    wiki_tables = [pq.read_table(p, columns=["title", "text"]) for p in a.wiki]
    stop: set[str] = set()
    if not a.legacy:
        if Path(a.types).exists():
            wd_types = json.loads(Path(a.types).read_text())
            # entities with no P31 at all (many villages) keep the type of their relation slot
            no_class = {q for q, cs in json.loads(Path(a.classes).read_text()).items() if not cs} \
                if Path(a.classes).exists() else set()
            ent_type = {q: wd_types.get(q, t) for q, t in ent_type.items() if q in wd_types or q in no_class}
            print(f"typed {len(ent_type)} entities from Wikidata classes")
        # single-word names that are common words (மேலே "above" -> Malé) are not names.
        # A frequency cutoff also removed real short names (இந்திய, கோவை, நேரு), so
        # the list is reviewed by hand.
        stop = set() if "alias_stop" in a.skip_fix else {
            l.split("#")[0].strip() for l in open(a.alias_stoplist, encoding="utf-8")} - {""}
        alias_stop = {n[1:] for n in stop if n.startswith("~")}
        stop -= {"~" + n for n in alias_stop}
        ent_label = {**{f["s"]: f["s_label"] for f in facts}, **{f["o"]: f["o_label"] for f in facts}}
        n_drop = 0
        for q in ent_names:
            keep = {n for n in ent_names[q] - stop if n not in alias_stop or n == ent_label.get(q)}
            n_drop += len(ent_names[q]) - len(keep); ent_names[q] = keep
        print(f"dropped {n_drop} stoplisted aliases")
    # given names: non-final words of multi-word person names, minus titles,
    # kinship words and regnal ordinals (சர், நடிகை, தம்பி, முதலாம்), which come
    # before a full name without making it a surname
    person_words = {w for q, ns in ent_names.items() if ent_type.get(q) == "PERSON"
                    for n in ns for w in WORD_RE.findall(n)[:-1]
                    if len(w) >= 3 and w not in TITLE_WORDS and not w.endswith("ாம்")}
    rel_types = {(r["subj"], r["obj"]) for r in rels.values()}

    min_inflect = 0 if a.legacy or "short_inflect" in a.skip_fix else 4
    gaz = Gazetteer(min_inflect)
    for q, names in ent_names.items():
        for n in names:
            gaz.add(n, q)
    # subject short forms (last word of a multi-word person name, e.g. பாரதி)
    subj_gaz_extra: dict[str, set[str]] = defaultdict(set)
    for q in subj_facts:
        if ent_type.get(q) == "PERSON":
            for n in list(ent_names[q]):
                w = WORD_RE.findall(n)
                if len(w) > 1 and len(w[-1]) >= 4 and (a.legacy or w[-1] not in stop):
                    subj_gaz_extra[q].add(w[-1])

    # which object types can pair with a subject type
    pair_types = defaultdict(set)
    for r in rels.values():
        pair_types[r["subj"]].add(r["obj"])

    stats = Counter()
    n_out = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for table in wiki_tables:
            for title, text in zip(table.column("title").to_pylist(), table.column("text").to_pylist()):
                if only_docs is not None and title not in only_docs:
                    continue
                sq = subj_title.get(title)
                if sq is None or sq not in ent_type:
                    continue
                stats["articles"] += 1
                s_gaz = Gazetteer(min_inflect)
                for n in ent_names[sq] | subj_gaz_extra[sq]:
                    s_gaz.add(n, sq)
                negs = []
                for si, sent in enumerate(split_sentences(text)):
                    words = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(sent)]
                    if not (a.min_words <= len(words) <= a.max_words):
                        continue
                    s_hits = s_gaz.find(words, sent)
                    o_hits = gaz.find(words, sent)
                    if not a.legacy:
                        own = {w for n in ent_names[sq] for w in WORD_RE.findall(n)}
                        def ok(h, q):
                            pw, nw = prev_word(words, h[0]), next_word(words, h[1])
                            # a fragment (fewer words than the label) right after someone
                            # else's given name, with nothing but a space between
                            fragment = len(WORD_RE.findall(sent[h[0]:h[1]])) < len(WORD_RE.findall(ent_label.get(q, "")))
                            adjacent = sent[:h[0]].rstrip() != sent[:h[0]] and sent[:h[0]].rstrip().endswith(pw)
                            if ("surname" not in a.skip_fix and ent_type.get(q) == "PERSON" and fragment
                                    and adjacent and pw in person_words and pw not in own):
                                return False      # surname of someone else: சஞ்சய் [காந்தி]
                            if ("institution" not in a.skip_fix and ent_type.get(q) != "PERSON"
                                    and nw.startswith(INSTITUTION_HEADS)):
                                return False      # part of an institution name
                            return True
                        allh = s_hits + o_hits if "nested" in a.skip_fix else drop_nested(s_hits + o_hits)
                        s_hits = [h for h in s_hits if h in allh and ok(h, sq)]
                        o_hits = [h for h in o_hits if h in allh and ok(h, h[2])]
                    if not s_hits:
                        continue
                    s0, s1, _, s_match = max(s_hits, key=lambda h: h[1] - h[0])
                    if not a.legacy and "initials" not in a.skip_fix and ent_type[sq] == "PERSON":
                        s0 = extend_initials(sent, extend_own(sent, s0, name_pairs(ent_names[sq])))
                    seen = set()
                    for o0, o1, oq, o_match in o_hits:
                        if not a.legacy and "initials" not in a.skip_fix and ent_type.get(oq) == "PERSON":
                            o0 = extend_initials(sent, extend_own(sent, o0, name_pairs(ent_names[oq])))
                        if oq == sq or oq in seen or not (o1 <= s0 or o0 >= s1):
                            continue
                        seen.add(oq)
                        if ent_type.get(oq) not in pair_types[ent_type[sq]]:
                            continue
                        labels = sorted(subj_facts[sq].get(oq, ()))
                        if not a.legacy and "type_check" not in a.skip_fix:
                            kept = [l for l in labels if (rels[l]["subj"], rels[l]["obj"]) == (ent_type[sq], ent_type[oq])]
                            stats["type_mismatch_dropped"] += len(labels) > 0 and not kept
                            if labels and not kept:
                                continue
                            labels = kept
                        ex = {
                            "sentence": sent,
                            "entities": [
                                {"id": "E1", "text": sent[s0:s1], "type": ent_type[sq], "start": s0, "end": s1},
                                {"id": "E2", "text": sent[o0:o1], "type": ent_type[oq], "start": o0, "end": o1},
                            ],
                            "relation": {"subject": "E1", "label": labels[0] if labels else NO_RELATION,
                                         "object": "E2"},
                            "variety": None,
                            "source_id": f"tawiki:{title}#s{si}",
                            "meta": {"ds_labels": labels, "wd_subject": sq, "wd_object": oq,
                                     "match": [s_match, o_match], "doc": f"tawiki:{title}",
                                     "verified": False},
                        }
                        if labels:
                            fo.write(json.dumps(ex, ensure_ascii=False) + "\n"); n_out += 1
                            stats["pos"] += 1; stats["pos:" + labels[0]] += 1
                            stats["multi_label"] += len(labels) > 1
                            stats["inflected_obj"] += o_match == "inflected"
                        else:
                            negs.append(ex)
                for ex in rng.sample(negs, min(len(negs), a.neg_per_article)):
                    fo.write(json.dumps(ex, ensure_ascii=False) + "\n"); n_out += 1
                    stats["neg"] += 1
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=1))
    print(f"wrote {n_out} candidates to {a.out}")


if __name__ == "__main__":
    main()
