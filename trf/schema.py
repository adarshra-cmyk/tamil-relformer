"""Dataset schema (plan section 5.2) plus character offsets.

The plan's example stores only entity text. Text alone is ambiguous when a
string occurs twice in a sentence, so every entity also carries `start`/`end`
character offsets into `sentence`. `load_jsonl` fills offsets from text when
they are missing and the match is unique, and refuses otherwise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

NO_RELATION = "NO_RELATION"
ENTITY_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "DATE", "MISC"}
TYPE_ABBR = {"PERSON": "PER", "LOCATION": "LOC", "ORGANIZATION": "ORG", "DATE": "DATE", "MISC": "MISC"}


@dataclass
class Entity:
    id: str
    text: str
    type: str
    start: int
    end: int


@dataclass
class Example:
    sentence: str
    entities: list[Entity]
    subject: str
    object: str
    label: str
    variety: str | None = None
    source_id: str | None = None
    meta: dict = field(default_factory=dict)

    def ent(self, eid: str) -> Entity:
        return next(e for e in self.entities if e.id == eid)

    @property
    def subj(self) -> Entity:
        return self.ent(self.subject)

    @property
    def obj(self) -> Entity:
        return self.ent(self.object)

    def to_json(self) -> dict:
        return {
            "sentence": self.sentence,
            "entities": [asdict(e) for e in self.entities],
            "relation": {"subject": self.subject, "label": self.label, "object": self.object},
            "variety": self.variety,
            "source_id": self.source_id,
            **({"meta": self.meta} if self.meta else {}),
        }


class SchemaError(ValueError):
    pass


def _resolve_offsets(sentence: str, e: dict) -> tuple[int, int]:
    if "start" in e and "end" in e:
        s, t = int(e["start"]), int(e["end"])
        if sentence[s:t] != e["text"]:
            raise SchemaError(f"offset mismatch for {e['id']}: {sentence[s:t]!r} != {e['text']!r}")
        return s, t
    text = e["text"].strip()
    first = sentence.find(text)
    if first < 0:
        raise SchemaError(f"entity {e['id']} text {text!r} not in sentence")
    if sentence.find(text, first + 1) >= 0:
        raise SchemaError(f"entity {e['id']} text {text!r} occurs more than once; give offsets")
    return first, first + len(text)


def parse_example(d: dict) -> Example:
    sent = d["sentence"]
    ents = []
    for e in d["entities"]:
        if e["type"] not in ENTITY_TYPES:
            raise SchemaError(f"unknown entity type {e['type']!r}")
        s, t = _resolve_offsets(sent, e)
        ents.append(Entity(e["id"], sent[s:t], e["type"], s, t))
    ids = {e.id for e in ents}
    rel = d["relation"]
    if rel["subject"] not in ids or rel["object"] not in ids or rel["subject"] == rel["object"]:
        raise SchemaError(f"bad relation endpoints {rel}")
    return Example(sent, ents, rel["subject"], rel["object"], rel["label"],
                   d.get("variety"), d.get("source_id"), d.get("meta", {}))


def load_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(parse_example(json.loads(line)))
            except (SchemaError, KeyError) as err:
                raise SchemaError(f"{path}:{i}: {err}") from err
    return out


def save_jsonl(examples: list[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
