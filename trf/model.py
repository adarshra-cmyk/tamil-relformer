"""Entity-pair relation classifier.

pair = [h_subj_marker ; h_obj_marker]                       (E0, E1, E2a)
pair = [h_subj_marker ; h_obj_marker ; e(case_s) ; e(case_o)]  (E2b)
logits = MLP(pair)

E4 (plan section 7, RQ3) adds a supervised contrastive loss on `pair`, optionally
through a projection head z = MLP_proj(pair) as in SupCon (Khosla et al. 2020).
The projection head is only built when asked for, so models without E4 have the
same parameters (and the same initialisation order) as before.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .morph import CASE_VOCAB


class RelationClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, vocab_size: int,
                 case_embedding: bool = False, case_dim: int = 32, dropout: float = 0.1,
                 encoder: nn.Module | None = None, proj_dim: int | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(vocab_size)
        h = self.encoder.config.hidden_size
        self.case_embedding = case_embedding
        in_dim = 2 * h
        if case_embedding:
            self.case_emb = nn.Embedding(len(CASE_VOCAB), case_dim)
            in_dim += 2 * case_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h, num_labels),
        )
        # SupCon projection head (2-layer MLP, ReLU, as in Khosla et al. 2020); built after
        # the classifier head so the head's initialisation does not change when it is added
        self.proj = None
        if proj_dim:
            self.proj = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim))

    def forward(self, input_ids, attention_mask, subj_pos, obj_pos, subj_case=None, obj_case=None,
                return_pair: bool = False):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = torch.arange(hs.size(0), device=hs.device)
        parts = [hs[idx, subj_pos], hs[idx, obj_pos]]
        if self.case_embedding:
            parts += [self.case_emb(subj_case), self.case_emb(obj_case)]
        pair = torch.cat(parts, dim=-1)
        logits = self.head(pair)
        if not return_pair:
            return logits
        # contrastive branch: the classifier input (before the head's dropout), or its projection
        return logits, (self.proj(pair) if self.proj is not None else pair)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1,
                anchor_ok: torch.Tensor | None = None) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020, L_out form) over one batch.

    z: (B, d) features (L2-normalised here). labels: (B,) gold label ids.
    anchor_ok: (B,) bool; examples that may act as anchors and as positives. Examples
    with anchor_ok=False still appear in every denominator, i.e. they are negatives only.
    For anchor i with positives P(i) = {p != i : y_p = y_i, anchor_ok[p]}:
        l_i = -1/|P(i)| sum_{p in P(i)} log( exp(z_i.z_p/tau) / sum_{a != i} exp(z_i.z_a/tau) )
    The loss is the mean of l_i over anchors with |P(i)| >= 1; a batch with no such anchor
    gives 0 (still connected to the graph so backward works).
    """
    B = z.size(0)
    if B < 2:
        return z.sum() * 0.0
    if anchor_ok is None:
        anchor_ok = torch.ones(B, dtype=torch.bool, device=z.device)
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T / tau
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask & anchor_ok[:, None] & anchor_ok[None, :]
    n_pos = pos.sum(1)
    has = n_pos > 0
    if not has.any():
        return z.sum() * 0.0
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(1)[has] / n_pos[has])
    return per_anchor.mean()
