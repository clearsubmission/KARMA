"""
KARMA-F: Knowledge Anchors with Future-Aware Encoder

New in KARMA-F vs KARMA-T:
- Autoregressive future context predictor
- Predicts likely future relations for each entity
- Combines past history (backward) with predicted future (forward)
- Better for forecasting tasks: model knows where entity is likely heading
- Future prediction is soft/probabilistic — no ground truth leakage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
    """Encodes past history — same as KARMA-T."""
    def __init__(self, embedding_dim, num_anchors):
        super().__init__()
        self.d = embedding_dim
        self.K = num_anchors

        self.rel_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.time_proj = nn.Linear(1, embedding_dim)
        self.nbr_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.attn_q    = nn.Linear(embedding_dim, embedding_dim)
        self.attn_k    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.attn_v    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, num_anchors * embedding_dim)
        self.norm      = nn.LayerNorm(embedding_dim)

    def forward(self, rel_embs, nbr_embs, time_deltas, mask):
        B, L, d = rel_embs.shape
        t_enc  = torch.tanh(self.time_proj(time_deltas))
        r_enc  = torch.relu(self.rel_proj(rel_embs))
        n_enc  = torch.relu(self.nbr_proj(nbr_embs))
        events = torch.cat([r_enc, n_enc, t_enc], dim=-1)

        denom  = mask.sum(dim=1, keepdim=True).clamp(min=1)
        query  = (events * mask.unsqueeze(-1)).sum(dim=1) / denom
        Q = self.attn_q(query[:, :d]).unsqueeze(1)
        K = self.attn_k(events)
        V = self.attn_v(events)

        attn = (Q * K).sum(dim=-1) / (d ** 0.5)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn.unsqueeze(-1) * V).sum(dim=1)
        return self.norm(self.out_proj(out).view(B, self.K, d))


class FutureContextPredictor(nn.Module):
    """
    KARMA-F core: predicts likely future relations for an entity.

    Given entity's past history embedding, predict:
    1. Which relations are likely to appear in the future
    2. How far in the future (temporal horizon)
    3. What kind of entities it will interact with

    This is a SOFT prediction — no ground truth future used.
    The future context is derived purely from past patterns.
    """
    def __init__(self, embedding_dim, num_relations, num_anchors,
                 future_steps=3):
        super().__init__()
        self.d            = embedding_dim
        self.R            = num_relations
        self.K            = num_anchors
        self.future_steps = future_steps

        # Predict future relation distribution from past embedding
        self.relation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_relations),
        )

        # Predict temporal horizon (how far into future)
        self.horizon_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, future_steps),
            nn.Softmax(dim=-1)
        )

        # Encode predicted future into embedding space
        self.future_encoder = nn.Sequential(
            nn.Linear(embedding_dim + num_relations, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_anchors * embedding_dim)
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, past_emb, tau):
        """
        past_emb: (B, d) entity embedding from past history
        tau:      (B,)   current query timestamp

        Returns: (B, K, d) future-augmented anchor representations
        """
        B, d = past_emb.shape

        # Predict future relation distribution (soft, no ground truth)
        rel_logits  = self.relation_predictor(past_emb)          # (B, R)
        rel_probs   = F.softmax(rel_logits, dim=-1)               # (B, R)

        # Predict temporal horizon weights
        horizon_w   = self.horizon_predictor(past_emb)            # (B, S)

        # Future context = weighted combination of predicted relations
        # scaled by temporal horizon
        future_ctx  = torch.cat([past_emb, rel_probs], dim=-1)   # (B, d+R)

        # Encode future context into anchor space
        future_repr = self.future_encoder(future_ctx)             # (B, K*d)
        future_repr = future_repr.view(B, self.K, d)             # (B, K, d)

        return future_repr, rel_probs


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult',
                 max_history=20, future_steps=3, future_weight=0.3):
        super().__init__()
        self.num_entities   = num_entities
        self.num_relations  = num_relations
        self.d              = embedding_dim
        self.K              = num_anchors
        self.sigma          = sigma
        self.score_func     = score_func
        self.max_history    = max_history
        self.future_weight  = future_weight

        self.anchor_times = nn.Parameter(torch.randn(num_anchors))

        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)

        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # Past history encoder (same as KARMA-T)
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-F: future context predictor
        self.future_predictor = FutureContextPredictor(
            embedding_dim, num_relations, num_anchors, future_steps)

        # Learned gate: how much to trust future vs past
        self.future_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid()
        )

        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def anchor_attention(self, tau):
        diff  = tau.unsqueeze(1) - self.anchor_times.unsqueeze(0)
        return F.softmax(-diff.pow(2) / (self.sigma + 1e-8), dim=-1)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau):
        """Past history encoding + future context augmentation."""
        B, L = rel_ids.shape
        dev  = rel_ids.device

        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)

        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()
        has_history = (lengths > 0).float().unsqueeze(-1)

        # Past encoding
        anchor_reps_past = self.encoder(
            rel_embs, nbr_embs, time_deltas, mask)                # (B, K, d)

        alpha    = self.anchor_attention(tau)                     # (B, K)
        past_emb = (alpha.unsqueeze(-1) * anchor_reps_past).sum(dim=1)  # (B, d)

        # Future prediction from past context
        anchor_reps_future, _ = self.future_predictor(
            past_emb, tau)                                         # (B, K, d)
        future_emb = (alpha.unsqueeze(-1) *
                      anchor_reps_future).sum(dim=1)              # (B, d)

        # Learned gate: trust future more when history is rich
        gate = self.future_gate(
            torch.cat([past_emb, future_emb], dim=-1))            # (B, 1)
        combined_emb = ((1 - gate) * past_emb +
                        gate * future_emb)                        # (B, d)

        # Cold start fallback
        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        return has_history * combined_emb + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        """KARMA-F hybrid embedding."""
        ind_emb = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau)
        ind_emb = self.layer_norm(self.dropout(ind_emb))

        if entity_ids is None:
            return ind_emb

        trans_emb = self.layer_norm(
            self.dropout(self.entity_table(entity_ids)))

        hist_ratio = (lengths.float() / max(self.max_history, 1)).clamp(0, 1)
        gate_input = torch.cat([
            ind_emb, trans_emb,
            hist_ratio.unsqueeze(-1)
        ], dim=-1)
        alpha_gate = self.gate_net(gate_input)

        hybrid = alpha_gate * ind_emb + (1.0 - alpha_gate) * trans_emb
        return self.layer_norm(hybrid)

    def future_relation_loss(self, entity_ids, rel_ids, taus):
        """
        Auxiliary loss: predicted future relations should match
        actual relations that appear after the current timestamp.
        Uses the current batch's relations as soft supervision.
        """
        # Get entity past embeddings
        # Use entity_table as proxy for current state
        past_emb   = self.layer_norm(self.entity_table(entity_ids))
        rel_logits, _ = self.future_predictor.relation_predictor(
            past_emb), None
        rel_logits = self.future_predictor.relation_predictor(past_emb)

        # Cross entropy: predicted relations should match actual relations
        loss = F.cross_entropy(rel_logits, rel_ids)
        return loss

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus, s_eids=None, o_eids=None):

        s_emb = self.entity_embed(s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_emb = self.entity_embed(o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])

        pos_scores = (s_emb * r_emb * o_emb).sum(dim=-1)
        return pos_scores
