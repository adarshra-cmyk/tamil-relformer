"""Fetch Wikidata facts for distant supervision.

Step 1 (separate, one query): data/processed/tawiki_qids.tsv maps every Tamil
Wikipedia page to its Wikidata item.
Step 2 (this script): for main-namespace articles present in the Wikipedia
dump, query the properties in configs/relations.json in batches of subject
ids (per-property queries over all of Wikidata time out for P27 etc.).
Objects must have a Tamil label. Writes data/processed/wikidata_facts.jsonl:
    {"rel": "PER:BIRTH_PLACE", "pid": "P19", "s": "Q..", "s_label": "..",
     "s_title": "<tawiki title>", "o": "Q..", "o_label": "..", "o_aliases": [..]}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "TamilRelFormer-research/0.1 (academic relation-extraction dataset construction)"

QUERY = """
SELECT ?s ?p ?sLabel ?o ?oLabel (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
  VALUES ?s {{ {subjects} }}
  VALUES ?p {{ {props} }}
  ?s ?p ?o .
  ?o rdfs:label ?oLabel . FILTER(LANG(?oLabel) = "ta")
  OPTIONAL {{ ?s rdfs:label ?sLabel . FILTER(LANG(?sLabel) = "ta") }}
  OPTIONAL {{ ?o skos:altLabel ?alias . FILTER(LANG(?alias) = "ta") }}
}}
GROUP BY ?s ?p ?sLabel ?o ?oLabel
"""


def run_query(q: str, retries: int = 5) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, data={"query": q, "format": "json"},
                              headers={"User-Agent": UA}, timeout=120)
        except requests.RequestException as e:
            print(f"  {e}; retry", file=sys.stderr); time.sleep(20); continue
        if r.status_code == 200:
            return r.json()["results"]["bindings"]
        wait = int(r.headers.get("Retry-After", 20 * (attempt + 1)))
        print(f"  HTTP {r.status_code}; retry in {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError("query failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", default="configs/relations.json")
    ap.add_argument("--qids", default="data/processed/tawiki_qids.tsv")
    ap.add_argument("--wiki", nargs="+", default=sorted(map(str, Path("data/raw/tawiki").glob("*.parquet"))))
    ap.add_argument("--out", default="data/processed/wikidata_facts.jsonl")
    ap.add_argument("--batch", type=int, default=800)
    a = ap.parse_args()

    rels = json.loads(Path(a.relations).read_text(encoding="utf-8"))["relations"]
    pid2rel = {pid: r["label"] for r in rels for pid in r["wikidata"]}
    titles = set()
    for p in a.wiki:
        titles.update(pq.read_table(p, columns=["title"]).column("title").to_pylist())
    q2title = {}
    for line in open(a.qids, encoding="utf-8").readlines()[1:]:
        s, t = line.rstrip("\n").split("\t")
        t = t.rsplit("@", 1)[0].strip('"')
        if t in titles:
            q2title[s.strip("<>").rsplit("/", 1)[-1]] = t
    qs = sorted(q2title)
    print(f"{len(qs)} subjects with articles in the dump", flush=True)

    props = " ".join(f"wdt:{p}" for p in pid2rel)
    tail = lambda uri: uri.rsplit("/", 1)[-1]
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(0, len(qs), a.batch):
            chunk = qs[i:i + a.batch]
            rows = run_query(QUERY.format(subjects=" ".join(f"wd:{q}" for q in chunk), props=props))
            for b in rows:
                s, pid = tail(b["s"]["value"]), tail(b["p"]["value"])
                f.write(json.dumps({
                    "rel": pid2rel[pid], "pid": pid, "s": s,
                    "s_label": b.get("sLabel", {}).get("value", q2title[s]),
                    "s_title": q2title[s], "o": tail(b["o"]["value"]),
                    "o_label": b["oLabel"]["value"],
                    "o_aliases": [x for x in b.get("aliases", {}).get("value", "").split("|") if x],
                }, ensure_ascii=False) + "\n")
            n += len(rows)
            print(f"  {i + len(chunk)}/{len(qs)} subjects, {n} facts", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
