"""E4: supervised contrastive loss (trf.model.supcon_loss) and its opt-in wiring in trf.train."""
import json
import math

import pytest
import torch
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

from trf.model import supcon_loss
from test_core import SAM


def reference_supcon(z, y, tau, ok):
    """Direct loop over the SupCon L_out formula."""
    z = torch.nn.functional.normalize(z, dim=-1)
    B, losses = len(y), []
    for i in range(B):
        if not ok[i]:
            continue
        P = [p for p in range(B) if p != i and y[p] == y[i] and ok[p]]
        if not P:
            continue
        denom = sum(math.exp(float(z[i] @ z[a]) / tau) for a in range(B) if a != i)
        losses.append(-sum(math.log(math.exp(float(z[i] @ z[p]) / tau) / denom) for p in P) / len(P))
    return sum(losses) / len(losses) if losses else 0.0


def test_closed_form_two_positives_one_negative():
    # z0 = z1 (same label), z2 orthogonal: each of 0, 1 has one positive (sim 1) and one negative (sim 0)
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0, 1])
    tau = 0.5
    expected = -math.log(math.exp(2.0) / (math.exp(2.0) + 1.0))
    assert supcon_loss(z, y, tau).item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_with_and_without_anchor_mask(seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(10, 6, generator=g)
    y = torch.randint(0, 3, (10,), generator=g)
    ok_all = torch.ones(10, dtype=torch.bool)
    ok = y != 0                                   # label 0 plays NO_RELATION: negative only
    for mask in (ok_all, ok):
        assert supcon_loss(z, y, 0.1, mask).item() == pytest.approx(reference_supcon(z, y, 0.1, mask), rel=1e-5)


def test_no_positives_gives_zero_and_backward_works():
    z = torch.randn(4, 3, requires_grad=True)
    loss = supcon_loss(z, torch.tensor([0, 1, 2, 3]), 0.1)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None and torch.all(z.grad == 0)


def test_gradient_pulls_positives_together():
    z = torch.randn(6, 4, generator=torch.Generator().manual_seed(3)).requires_grad_()
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(z, y, 0.1)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    with torch.no_grad():       # a small step against the gradient lowers the loss
        assert supcon_loss(z - 1e-2 * z.grad, y, 0.1) < loss


@pytest.fixture(scope="module")
def tok():
    from trf.encode import add_marker_tokens
    t = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    add_marker_tokens(t)
    return t


def _toy(tmp_path):
    rows = []
    for i in range(12):
        d = json.loads(json.dumps(SAM)); d["source_id"] = f"t{i}"
        if i % 2:
            d["sentence"] = "சாம் யாழ்ப்பாணத்தில் பேசினார்."
            d["relation"]["label"] = "NO_RELATION"
        rows.append(d)
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def _run(tmp_path, tok, name, **extra):
    from trf.train import DEFAULTS, run
    cfg = {**DEFAULTS, "name": name, "case_embedding": True, "max_steps": 4, "eval_every": 2,
           "batch_size": 4, "device": "cpu", **{s: str(tmp_path / f"{s}.jsonl") for s in ("train", "dev", "test")},
           **extra}
    torch.manual_seed(0)    # same random tiny encoder for every run
    enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tok) + 64, hidden_size=32, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=64))
    return run(cfg, seed=1, train_limit=None, out_root=str(tmp_path / "runs"),
               encoder_override=enc, tokenizer_override=tok)


def test_lambda_zero_equals_baseline(tmp_path, tok):
    """supcon_lambda=0 (with the other supcon keys set) reproduces the run without any supcon key."""
    _toy(tmp_path)
    base = _run(tmp_path, tok, "base")
    off = _run(tmp_path, tok, "off", supcon_lambda=0, supcon_tau=0.1, supcon_proj=True)
    assert [h["train_loss"] for h in base["history"]] == [h["train_loss"] for h in off["history"]]
    assert [h["dev_micro_f1"] for h in base["history"]] == [h["dev_micro_f1"] for h in off["history"]]
    assert all("train_supcon" not in h for h in off["history"])


@pytest.mark.parametrize("proj,norel", [(True, "negative"), (False, "class")])
def test_supcon_run_completes(tmp_path, tok, proj, norel):
    _toy(tmp_path)
    res = _run(tmp_path, tok, f"sc{proj}{norel}", supcon_lambda=0.5, supcon_proj=proj, supcon_norel=norel)
    assert "micro_f1" in res["test"]
    assert all(math.isfinite(h["train_supcon"]) for h in res["history"])
    base = _run(tmp_path, tok, f"b{proj}{norel}")
    assert [h["train_loss"] for h in res["history"]] != [h["train_loss"] for h in base["history"]]
