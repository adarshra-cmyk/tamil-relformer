import json

import pytest
import torch
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

from trf.encode import MorphConfig, add_marker_tokens, encode, marked_text
from trf.metrics import evaluate
from trf.schema import SchemaError, parse_example

SAM = {
    "sentence": "சாம் யாழ்ப்பாணத்தில் வாழ்ந்தார்.",
    "entities": [{"id": "E1", "text": "சாம்", "type": "PERSON"},
                 {"id": "E2", "text": "யாழ்ப்பாணத்தில்", "type": "LOCATION"}],
    "relation": {"subject": "E1", "label": "PER:RESIDENCE", "object": "E2"},
    "meta": {"morph_pred": {"subj": {"lemma": "சாம்", "case": "Nom"},
                            "obj": {"lemma": "யாழ்ப்பாணம்", "case": "Loc"}}},
}


def test_offsets_filled_from_text():
    ex = parse_example(SAM)
    assert ex.sentence[ex.obj.start:ex.obj.end] == "யாழ்ப்பாணத்தில்"


def test_duplicate_text_needs_offsets():
    d = json.loads(json.dumps(SAM))
    d["sentence"] = "சாம் சாம் யாழ்ப்பாணத்தில் வாழ்ந்தார்."
    with pytest.raises(SchemaError):
        parse_example(d)


def test_marker_variants():
    ex = parse_example(SAM)
    assert marked_text(ex, MorphConfig()) == \
        "<SUBJ-PER> சாம் </SUBJ-PER> <OBJ-LOC> யாழ்ப்பாணத்தில் </OBJ-LOC> வாழ்ந்தார்."
    assert "<CASE-Loc> யாழ்ப்பாணத்தில்" in marked_text(ex, MorphConfig(case_in_text=True))
    assert "யாழ்ப்பாணத்தில் <LEM> யாழ்ப்பாணம் </OBJ-LOC>" in marked_text(ex, MorphConfig(lemma_in_text=True))


def test_object_before_subject():
    d = json.loads(json.dumps(SAM))
    d["relation"] = {"subject": "E2", "label": "X", "object": "E1"}
    t = marked_text(parse_example(d), MorphConfig())
    assert t.startswith("<OBJ-PER> சாம் </OBJ-PER> <SUBJ-LOC>")


def test_metrics_exclude_no_relation():
    m = evaluate(["A", "NO_RELATION", "B", "NO_RELATION"], ["A", "A", "NO_RELATION", "NO_RELATION"])
    assert m["micro_p"] == pytest.approx(0.5) and m["micro_r"] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")


def test_marker_positions(tok):
    add_marker_tokens(tok)
    ex = parse_example(SAM)
    e = encode(ex, tok, MorphConfig(case_in_text=True), {"PER:RESIDENCE": 0})
    ids = e["input_ids"]
    assert tok.convert_ids_to_tokens(ids[e["subj_pos"]]) == "<SUBJ-PER>"
    assert tok.convert_ids_to_tokens(ids[e["obj_pos"]]) == "<OBJ-LOC>"


def test_end_to_end_tiny(tmp_path, tok):
    """Train a tiny random encoder on a toy set to check the loop runs and writes outputs."""
    from trf.train import DEFAULTS, run
    rows = []
    for i in range(12):
        d = json.loads(json.dumps(SAM)); d["source_id"] = f"t{i}"
        if i % 2:
            d["sentence"] = "சாம் யாழ்ப்பாணத்தில் பேசினார்."
            d["relation"]["label"] = "NO_RELATION"
        rows.append(d)
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    cfg = {**DEFAULTS, "name": "tiny", "case_embedding": True, "epochs": 2, "batch_size": 4,
           **{s: str(tmp_path / f"{s}.jsonl") for s in ("train", "dev", "test")}}
    enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tok) + 64, hidden_size=32, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=64))
    res = run(cfg, seed=1, train_limit=8, out_root=str(tmp_path / "runs"),
              encoder_override=enc, tokenizer_override=tok)
    assert "micro_f1" in res["test"]
    assert (tmp_path / "runs/tiny/seed1_n8/metrics.json").exists()


def test_type_mask_blocks_wrong_signature(tmp_path, tok):
    """With type_mask, a PERSON/LOCATION pair can never be predicted as PER:SPOUSE."""
    from trf.train import DEFAULTS, run
    rows = []
    for i in range(12):
        d = json.loads(json.dumps(SAM)); d["source_id"] = f"t{i}"
        if i % 3 == 0:   # a PERSON/PERSON spouse item so PER:SPOUSE is in the label set
            d["sentence"] = "சாம் ராணியை மணந்தார்."
            d["entities"] = [{"id": "E1", "text": "சாம்", "type": "PERSON"},
                             {"id": "E2", "text": "ராணியை", "type": "PERSON"}]
            d["relation"]["label"] = "PER:SPOUSE"
        rows.append(d)
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    cfg = {**DEFAULTS, "name": "tm", "type_mask": True, "epochs": 1, "max_steps": 4, "eval_every": 2,
           "batch_size": 4, **{s: str(tmp_path / f"{s}.jsonl") for s in ("train", "dev", "test")}}
    enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tok) + 64, hidden_size=32, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=64))
    run(cfg, seed=1, train_limit=None, out_root=str(tmp_path / "runs"), encoder_override=enc, tokenizer_override=tok)
    preds = [json.loads(l) for l in open(tmp_path / "runs/tm/seed1/test_predictions.jsonl")]
    for r, p in zip(rows, preds):
        if r["entities"][1]["type"] == "LOCATION":
            assert p["pred"] != "PER:SPOUSE"


def test_suffix_mask_runs(tmp_path, tok):
    """E3 masking: suffix positions are found and a masked training run completes (and its control)."""
    from trf.encode import encode, MorphConfig
    from trf.train import DEFAULTS, run
    d = json.loads(json.dumps(SAM))
    d["meta"] = {"morph_pred": {"subj": {"lemma": "சாம்", "case": "Nom"}, "obj": {"lemma": "யாழ்ப்பாணம்", "case": "Loc"}}}
    ex = parse_example(d)
    e = encode(ex, tok, MorphConfig(), {ex.label: 0}, with_suffix=True)
    assert e["suffix_pos"], "the locative suffix of யாழ்ப்பாணத்தில் should be found"
    rows = []
    for i in range(8):
        r = json.loads(json.dumps(d)); r["source_id"] = f"t{i}"; rows.append(r)
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    for ctrl in (None, "random"):
        cfg = {**DEFAULTS, "name": f"m{ctrl}", "suffix_mask_prob": 1.0, "mask_control": ctrl, "max_steps": 2,
               "eval_every": 2, "batch_size": 4, **{s: str(tmp_path / f"{s}.jsonl") for s in ("train", "dev", "test")}}
        enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tok) + 64, hidden_size=32, num_hidden_layers=1,
                                               num_attention_heads=2, intermediate_size=64))
        res = run(cfg, seed=1, train_limit=None, out_root=str(tmp_path / "runs"), encoder_override=enc, tokenizer_override=tok)
        assert "micro_f1" in res["test"]
