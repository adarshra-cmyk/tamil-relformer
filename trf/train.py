"""Train and evaluate one run (plan section 7).

    python -m trf.train --config configs/e0_baseline.json --seed 13
    python -m trf.train --config configs/e2b_case_emb.json --train-limit 500 --seed 13

Each run writes to runs/<name>/seed<k>[_n<limit>]/: the resolved config,
the hashes of the split files, per-epoch dev metrics, test metrics of the
best-dev epoch, and test predictions. Model weights are not saved unless
--save-model is given.
"""
from __future__ import annotations

import os

# Cap the MPS allocator: without this its cache grew to ~23 GB on a 26 GB
# machine and the run spent most of its time swapping.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .encode import MARKER_TOKENS, MorphConfig, add_marker_tokens, encode, marker_descriptions
from .metrics import evaluate
from .model import RelationClassifier, supcon_loss
from .schema import NO_RELATION, Example, load_jsonl

DEFAULTS = {
    "name": "e0_baseline",
    "encoder": "xlm-roberta-base",
    "train": "data/processed/train.jsonl",
    "dev": "data/processed/dev.jsonl",
    "test": "data/processed/test.jsonl",
    "lemma_in_text": False,
    "case_in_text": False,
    "case_embedding": False,
    "epochs": 5,           # used only when max_steps is null
    "max_steps": 1500,     # same number of updates for every train size (learning curves)
    "eval_every": 100,     # steps between dev evaluations
    "patience": 4,         # stop after this many evals without dev improvement (null = never)
    "min_delta": 0.005,    # dev micro-F1 gain that counts as improvement for patience
    "batch_size": 32,
    "lr": 3e-5,
    "head_lr": 1e-4,
    "warmup": 0.1,
    "weight_decay": 0.01,
    "max_len": 256,
    "grad_clip": 1.0,
}
# Optional keys read with cfg.get (absent = off, so older configs behave and log as before):
#   type_mask, suffix_mask_prob, mask_control  (see below)
#   device          "cpu" / "cuda" / "mps"; absent = pick_device()
#   marker_init     "informed" = new marker/case tokens start from description embeddings
#   E4, supervised contrastive objective (plan section 7, RQ3):
#   supcon_lambda   weight of SupCon added to CE; 0 or absent = off
#   supcon_tau      temperature (default 0.1)
#   supcon_proj     true = 2-layer MLP projection head to supcon_proj_dim (default 128) for the
#                   contrastive branch (SupCon); false = contrast the classifier input directly
#   supcon_norel    how NO_RELATION enters the loss: "negative" (default) or "class"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subsample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """Label-stratified subsample for learning curves; keeps every label present when n allows."""
    if n >= len(examples):
        return examples
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    for v in by_label.values():
        rng.shuffle(v)
    out = [v[0] for v in by_label.values()][:n]
    rest = [ex for v in by_label.values() for ex in v[1:]]
    rng.shuffle(rest)
    out += rest[: n - len(out)]
    rng.shuffle(out)
    return out


def collate(batch, pad_id, multiple: int = 32):
    # pad to a multiple of 32: MPS rebuilds kernels for every new shape, so
    # free-length batches are several times slower
    L = max(len(b["input_ids"]) for b in batch)
    L = -(-L // multiple) * multiple
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
        mask[i, : len(b["input_ids"])] = 1
    t = lambda k: torch.tensor([b[k] for b in batch], dtype=torch.long)
    out = {"input_ids": ids, "attention_mask": mask, "subj_pos": t("subj_pos"),
           "obj_pos": t("obj_pos"), "subj_case": t("subj_case"), "obj_case": t("obj_case"),
           "labels": t("label")}
    if "label_mask" in batch[0]:
        out["label_mask"] = torch.tensor([b["label_mask"] for b in batch], dtype=torch.bool)
    return out


@torch.no_grad()
def predict(model, loader, device) -> list[int]:
    model.eval()
    out = []
    for b in loader:
        labels = b.pop("labels")
        lm = b.pop("label_mask", None)
        logits = model(**{k: v.to(device) for k, v in b.items()})
        if lm is not None:
            logits = logits.masked_fill(~lm.to(device), -1e4)
        out += logits.argmax(-1).tolist()
    return out


def free_device_cache(device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def device_mem_gb(device) -> float | None:
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if device.type == "cuda":
        return round(torch.cuda.max_memory_reserved() / 1e9, 2)
    return None


def file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run(cfg: dict, seed: int, train_limit: int | None, out_root: str = "runs",
        save_model: bool = False, encoder_override=None, tokenizer_override=None,
        extra_tests: dict[str, str] | None = None) -> dict:
    set_seed(seed)
    device = torch.device(cfg["device"]) if cfg.get("device") else pick_device()
    mc = MorphConfig(cfg["lemma_in_text"], cfg["case_in_text"], cfg["case_embedding"])

    train = load_jsonl(cfg["train"]); dev = load_jsonl(cfg["dev"]); test = load_jsonl(cfg["test"])
    if train_limit:
        train = subsample(train, train_limit, seed)
    labels = sorted({ex.label for ex in train + dev + test} | {NO_RELATION})
    label2id = {l: i for i, l in enumerate(labels)}

    tok = tokenizer_override or AutoTokenizer.from_pretrained(cfg["encoder"])
    add_marker_tokens(tok)
    if cfg.get("type_mask"):
        # labels whose (subject, object) type signature fits the example's entity types;
        # NO_RELATION always allowed. Applied to the logits in training and prediction.
        sig = {r["label"]: (r["subj"], r["obj"])
               for r in json.loads(Path(cfg.get("relations", "configs/relations.json")).read_text())["relations"]}

        def allowed(x):
            t = (x.subj.type, x.obj.type)
            return [l == NO_RELATION or sig.get(l) == t for l in labels]

        enc = lambda xs: [{**encode(x, tok, mc, label2id, cfg["max_len"]), "label_mask": allowed(x)} for x in xs]
    else:
        enc = lambda xs: [encode(x, tok, mc, label2id, cfg["max_len"]) for x in xs]
    tr, dv, te = enc(train), enc(dev), enc(test)
    col = lambda b: collate(b, tok.pad_token_id)
    # E3, morphology-guided masking (plan section 7): during training only, each
    # suffix token of an entity is replaced by <mask> with probability suffix_mask_prob. The control
    # (mask_control="random") masks the same number of random sentence tokens instead.
    p_mask = cfg.get("suffix_mask_prob") or 0.0
    if p_mask:
        from .encode import encode as _encode
        for d, x in zip(tr, train):
            d["suffix_pos"] = _encode(x, tok, mc, label2id, cfg["max_len"], with_suffix=True)["suffix_pos"]
        special = set(tok.all_special_ids) | set(tok.convert_tokens_to_ids(MARKER_TOKENS))
        mask_rng = random.Random(seed + 7919)

        def masked(d):
            ids = list(d["input_ids"])
            pos = [i for i in d["suffix_pos"] if mask_rng.random() < p_mask]
            if pos and cfg.get("mask_control") == "random":
                pool = [i for i, t in enumerate(ids) if t not in special]
                pos = mask_rng.sample(pool, min(len(pos), len(pool)))
            for i in pos:
                ids[i] = tok.mask_token_id
            return {**d, "input_ids": ids}

        train_col = lambda b: collate([masked(d) for d in b], tok.pad_token_id)
    else:
        train_col = col
    tr_loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, collate_fn=train_col,
                           generator=torch.Generator().manual_seed(seed))
    dv_loader = DataLoader(dv, batch_size=64, collate_fn=col)
    te_loader = DataLoader(te, batch_size=64, collate_fn=col)

    # E4: loss = CE + supcon_lambda * SupCon(pair representation). Positives of an anchor are
    # the other in-batch examples with the same gold label. NO_RELATION handling
    # (supcon_norel): "negative" (default) means NO_RELATION examples are never anchors or
    # positives but stay in every denominator, so relation clusters are pushed away from
    # them while they are not pulled into one cluster themselves. NO_RELATION is a catch-all
    # for pairs that are unrelated for many different reasons, so forcing them together
    # imposes structure that has no single meaning; the CE term already separates them.
    # "class" treats NO_RELATION as an ordinary class (vanilla SupCon) for comparison.
    # Not implemented: plan's inflectional-variant positives from generated (ThamizhiMorph)
    # examples; every generated variant would need verification first (future work).
    sc_lambda = float(cfg.get("supcon_lambda") or 0.0)
    sc_tau = float(cfg.get("supcon_tau", 0.1))
    sc_norel = cfg.get("supcon_norel", "negative")
    if sc_norel not in ("negative", "class"):
        raise ValueError(f"supcon_norel must be 'negative' or 'class', got {sc_norel!r}")
    proj_dim = (cfg.get("supcon_proj_dim", 128) if sc_lambda and cfg.get("supcon_proj", True) else None)
    no_rel_id = label2id[NO_RELATION]
    model = RelationClassifier(cfg["encoder"], len(labels), len(tok), mc.case_embedding,
                               encoder=encoder_override, proj_dim=proj_dim).to(device)
    if cfg.get("marker_init") == "informed":
        # plan Step A ("a carefully tested marker scheme"): start each marker / case tag from
        # the mean embedding of its description (encode.marker_descriptions); deterministic
        emb = model.encoder.get_input_embeddings().weight
        with torch.no_grad():
            for t, text in marker_descriptions().items():
                ids = tok(text, add_special_tokens=False)["input_ids"]
                emb[tok.convert_tokens_to_ids(t)] = emb[ids].mean(0)
    elif cfg.get("marker_init") not in (None, "default"):
        raise ValueError(f"marker_init must be 'informed' or absent, got {cfg['marker_init']!r}")
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr"]},
                             {"params": other, "lr": cfg["head_lr"]}],
                            weight_decay=cfg["weight_decay"],
                            # foreach=True allocates parameter-sized temporaries (the
                            # 250k x 768 embedding is 736 MB) and ran out of MPS memory
                            foreach=False)
    steps = cfg.get("max_steps") or len(tr_loader) * cfg["epochs"]
    eval_every = cfg.get("eval_every") or len(tr_loader)
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    tag = f"seed{seed}" + (f"_n{train_limit}" if train_limit else "")
    out_dir = Path(out_root) / cfg["name"] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg, "seed": seed, "train_limit": train_limit, "n_train": len(train),
         "labels": labels, "morph": asdict(mc), "device": str(device),
         "split_sha256": {k: file_sha(cfg[k]) for k in ("train", "dev", "test")}},
        indent=2, ensure_ascii=False))

    gold_dev = [ex.label for ex in dev]
    history, best, best_state, bad = [], -1.0, None, 0

    def batches():
        while True:
            yield from tr_loader

    model.train(); t0 = time.time(); tot, n_loss, tot_sc = 0.0, 0, 0.0
    for step, b in enumerate(batches(), 1):
        labels_t = b.pop("labels").to(device)
        lm = b.pop("label_mask", None)
        if sc_lambda:
            logits, feats = model(**{k: v.to(device) for k, v in b.items()}, return_pair=True)
        else:
            logits = model(**{k: v.to(device) for k, v in b.items()})
        if lm is not None:
            logits = logits.masked_fill(~lm.to(device), -1e4)
        loss = loss_fn(logits, labels_t)
        if sc_lambda:
            anchor_ok = None if sc_norel == "class" else labels_t != no_rel_id
            sc = supcon_loss(feats, labels_t, sc_tau, anchor_ok)
            loss = loss + sc_lambda * sc
            tot_sc += sc.item()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step(); sched.step(); tot += loss.item(); n_loss += 1
        # the MPS cache otherwise grows past the watermark within a few hundred
        # steps; clearing every step kept the peak at ~9 GB with no measured slowdown
        free_device_cache(device)
        if step % eval_every and step != steps:
            continue
        dev_pred = [labels[i] for i in predict(model, dv_loader, device)]
        free_device_cache(device)
        m = evaluate(gold_dev, dev_pred, labels)
        history.append({"mem_gb": device_mem_gb(device), "step": step, "epoch": round(step / len(tr_loader), 2),
                        "train_loss": tot / n_loss, "dev_micro_f1": m["micro_f1"],
                        "dev_macro_f1": m["macro_f1"], "sec": round(time.time() - t0, 1)})
        if sc_lambda:   # train_loss above is the total (CE + lambda * SupCon)
            history[-1]["train_supcon"] = tot_sc / n_loss
        print(json.dumps(history[-1]), flush=True)
        model.train(); tot, n_loss, tot_sc = 0.0, 0, 0.0
        if m["micro_f1"] > best:
            bad = 0 if m["micro_f1"] >= best + cfg.get("min_delta", 0.0) else bad + 1
            best = m["micro_f1"]
            # reuse one CPU buffer: fresh .cpu() copies each eval left MPS staging
            # memory behind and the run went out of memory after a few evals
            if best_state is None:
                best_state = {k: torch.empty_like(v, device="cpu") for k, v in model.state_dict().items()}
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    best_state[k].copy_(v)
            free_device_cache(device)
        else:
            bad += 1
        if step >= steps or (cfg.get("patience") and bad >= cfg["patience"]):
            break

    model.load_state_dict(best_state)
    test_pred = [labels[i] for i in predict(model, te_loader, device)]
    test_m = evaluate([ex.label for ex in test], test_pred, labels)
    best_step = max(history, key=lambda h: h["dev_micro_f1"])["step"]
    result = {"history": history, "best_dev_micro_f1": best, "best_step": best_step,
              "stopped_step": history[-1]["step"], "test": test_m}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with open(out_dir / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for ex, p in zip(test, test_pred):
            f.write(json.dumps({"source_id": ex.source_id, "gold": ex.label, "pred": p,
                                "variety": ex.variety}, ensure_ascii=False) + "\n")
    # same best checkpoint on other versions of the test set (e.g. verified morphology)
    for name, path in (extra_tests or {}).items():
        xt = load_jsonl(path)
        xt_loader = DataLoader(enc(xt), batch_size=64, collate_fn=col)
        xt_pred = [labels[i] for i in predict(model, xt_loader, device)]
        xt_m = evaluate([ex.label for ex in xt], xt_pred, labels)
        result[f"test_{name}"] = xt_m
        with open(out_dir / f"test_predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for ex, p in zip(xt, xt_pred):
                f.write(json.dumps({"source_id": ex.source_id, "gold": ex.label, "pred": p,
                                    "variety": ex.variety}, ensure_ascii=False) + "\n")
        print(f"[{cfg['name']} {tag}] test_{name} micro-F1 {xt_m['micro_f1']:.4f}")
    if extra_tests:
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if save_model:
        torch.save(best_state, out_dir / "model.pt")
    print(f"[{cfg['name']} {tag}] test micro-F1 {test_m['micro_f1']:.4f} macro-F1 {test_m['macro_f1']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--extra-test", nargs="*", default=[], metavar="NAME=PATH",
                    help="also evaluate the best checkpoint on these test files")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (JSON values)")
    a = ap.parse_args()
    cfg = {**DEFAULTS, **json.loads(Path(a.config).read_text())}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    run(cfg, a.seed, a.train_limit, a.out, a.save_model,
        extra_tests=dict(x.split("=", 1) for x in a.extra_test))


if __name__ == "__main__":
    main()
