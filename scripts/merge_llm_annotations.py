"""Fill copies of the pilot sheets with LLM annotations (provisional, not human gold).

pass1 outputs fill sheet A and sheet B; for double-annotated items, sheet B
takes the independent pass2 label so `annotation_io.py agree` measures
model self-consistency. Writes pilot_sheet_{A,B}_llm.csv; the blank sheets
for human annotators are left untouched.
"""
import csv
import json
from pathlib import Path

base = Path("data/annotation/pilot_v0")
COLS = None
p1 = {}
for f in sorted((base / "llm").glob("out_pass1_b*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); p1[d["item_id"]] = d
p2 = {}
f2 = base / "llm/out_pass2_double.jsonl"
if f2.exists():
    for l in open(f2, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); p2[d["item_id"]] = d
FIELDS = ["label", "span_ok", "type_ok", "variety", "subj_lemma", "subj_case", "obj_lemma", "obj_case", "notes"]
missing = set()
for sheet in "AB":
    with open(base / f"pilot_sheet_{sheet}.csv", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f); COLS = rd.fieldnames; rows = list(rd)
    for r in rows:
        src = p2.get(r["item_id"]) if sheet == "B" and r["item_id"] in p2 else p1.get(r["item_id"])
        if src is None:
            missing.add(r["item_id"]); continue
        for k in FIELDS:
            r[k] = str(src.get(k, "") or "")
    with open(base / f"pilot_sheet_{sheet}_llm.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
print(f"pass1 {len(p1)} items, pass2 {len(p2)} items, missing {len(missing)}")
