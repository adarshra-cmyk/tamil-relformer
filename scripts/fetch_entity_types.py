"""Entity types from Wikidata classes, instead of from the relation slot.

The slot of LOC:LOCATED_IN / LOC:COUNTRY accepts any subject (temples,
festivals, parties, companies), so slot-derived types were often wrong in the
pilot. Here each entity gets a type from its P31 (instance of) classes:
    PERSON        instance of human (Q5), or, after LOCATION, of a subclass of
                  deity, mythical character, legendary figure or human who may
                  be fictional (Rama, Arjuna, Zeus)
    ORGANIZATION  instance of a subclass of organization (Q43229)
    LOCATION      instance of a subclass of geographic location (Q2221906),
                  or, failing all of the above, of architectural structure
                  (Q811979) or infrastructure (Q121359): temples, dams,
                  airports, bridges, stadiums
    (none)        anything else (languages, battles, elections, awards, lists)
When an entity falls in more than one (e.g. a city-state), the order above
decides. Writes data/processed/entity_types.json {qid: type}.
"""
import json
import sys
import time
from pathlib import Path

import requests

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "TamilRelFormer-research/0.1 (academic relation-extraction dataset construction)"
ROOTS = [("PERSON", "wd:Q5"), ("ORGANIZATION", "wd:Q43229"), ("LOCATION", "wd:Q2221906")]
# Countries, states and cities are also subclasses of organization in Wikidata
# (state -> political organization), so these geopolitical roots are checked
# before ORGANIZATION and map to LOCATION.
GEO_ROOTS = ["Q56061", "Q486972", "Q6256", "Q3624078", "Q82794", "Q7275"]
# Built structures have no geographic-location ancestor in Wikidata (13k Hindu
# temples were untyped). Checked last, so e.g. a university stays ORGANIZATION.
STRUCT_ROOTS = ["Q811979", "Q121359"]
# Mythological and legendary figures act like people in text (spouse, parent,
# birthplace). Checked after LOCATION so river goddesses (Ganga) stay places.
MYTH_ROOTS = ["Q178885", "Q4271324", "Q13002315", "Q21070568"]
Q_CLASSES = "SELECT ?e ?c WHERE {{ VALUES ?e {{ {vals} }} ?e wdt:P31 ?c . }}"
Q_PARENTS = "SELECT ?c ?p WHERE {{ VALUES ?c {{ {vals} }} ?c wdt:P279 ?p . }}"


def run(q, retries=6):
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, data={"query": q, "format": "json"},
                              headers={"User-Agent": UA}, timeout=120)
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
            wait = int(r.headers.get("Retry-After", 20 * (attempt + 1)))
            print(f"  HTTP {r.status_code}; retry in {wait}s", file=sys.stderr, flush=True)
        except requests.RequestException as e:
            wait = 20; print(f"  {e}; retry", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise RuntimeError("query failed")


tail = lambda b, k: b[k]["value"].rsplit("/", 1)[-1]


def main():
    facts = Path("data/processed/wikidata_facts.jsonl")
    cls_path = Path("data/processed/entity_classes.json")
    out = Path("data/processed/entity_types.json")
    qids = set()
    for l in open(facts, encoding="utf-8"):
        d = json.loads(l); qids.update((d["s"], d["o"]))
    qids = sorted(qids)

    # step 1: direct P31 classes per entity (cached)
    classes = json.loads(cls_path.read_text()) if cls_path.exists() else {}
    todo = [q for q in qids if q not in classes]
    for i in range(0, len(todo), 1000):
        chunk = todo[i:i + 1000]
        for q in chunk:
            classes[q] = []
        for b in run(Q_CLASSES.format(vals=" ".join(f"wd:{q}" for q in chunk))):
            classes[tail(b, "e")].append(tail(b, "c"))
        cls_path.write_text(json.dumps(classes))
        print(f"  classes {i + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(0.5)

    # step 2: walk superclass (P279) edges upward from the classes we have and
    # resolve roots locally. A downward P279* query from "organization" times out.
    parents_path = Path("data/processed/class_parents.json")
    parents = json.loads(parents_path.read_text()) if parents_path.exists() else {}
    frontier = sorted({c for cs in classes.values() for c in cs} - set(parents))
    level = 0
    while frontier and level < 25:
        for i in range(0, len(frontier), 500):
            chunk = frontier[i:i + 500]
            for c in chunk:
                parents.setdefault(c, [])
            for b in run(Q_PARENTS.format(vals=" ".join(f"wd:{c}" for c in chunk))):
                parents[tail(b, "c")].append(tail(b, "p"))
            time.sleep(0.3)
        parents_path.write_text(json.dumps(parents))
        frontier = sorted({p for c in frontier for p in parents[c]} - set(parents))
        level += 1
        print(f"  level {level}: {len(parents)} classes known, {len(frontier)} new", flush=True)

    root_ids = {name: r.split(":")[1] for name, r in ROOTS}
    memo: dict[str, set] = {}

    def roots_of(c, stack=()):
        if c in memo:
            return memo[c]
        if c in stack:
            return set()
        out = {name for name, rid in root_ids.items() if c == rid}
        if c in GEO_ROOTS:
            out.add("GEO")
        if c in STRUCT_ROOTS:
            out.add("STRUCT")
        if c in MYTH_ROOTS:
            out.add("MYTH")
        for p in parents.get(c, []):
            out |= roots_of(p, stack + (c,))
        memo[c] = out
        return out

    sys.setrecursionlimit(10000)
    under = {name: set() for name, _ in ROOTS}
    under["GEO"] = set()
    under["STRUCT"] = set()
    under["MYTH"] = set()
    for c in {c for cs in classes.values() for c in cs}:
        for name in roots_of(c):
            under[name].add(c)
    under["PERSON"] = {"Q5"}
    for name in under:
        print(f"  {name}: {len(under[name])} classes", flush=True)

    types = {}
    for q, cs in classes.items():
        for name in ("PERSON", "GEO", "ORGANIZATION", "LOCATION", "MYTH", "STRUCT"):   # precedence
            if any(c in under[name] for c in cs):
                types[q] = {"GEO": "LOCATION", "STRUCT": "LOCATION", "MYTH": "PERSON"}.get(name, name); break
    out.write_text(json.dumps(types))
    from collections import Counter
    print(Counter(types.values()), f"untyped {len(qids) - len(types)}")


if __name__ == "__main__":
    main()
