"""
KARMA-RL: KARMA with Soft Temporal Rule Integration

Adds learned temporal pattern scoring to KARMA-T.
Rules are NOT mined symbolically — they are learned
differentiably from the training data.

Pattern types learned:
1. Relation sequence: (r1,t-2) → (r2,t-1) → (r3,t)
2. Symmetric: (X,r,Y,t) → (Y,r_inv,X,t)
3. Temporal recurrence: (X,r,Y,t) → (X,r,Y,t+k)
4. Chain: (X,r1,Z,t) AND (Z,r2,Y,t) → (X,r3,Y,t)

Key advantage over symbolic rules:
- Works for UNSEEN entities (fires on new entities with history)
- Differentiable — end-to-end training
- Soft confidence — handles noisy patterns
- Generalizes to unseen timestamps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
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
        out  = (attn.unsqueeze(-1) * V).sum(dim=1)
        return self.norm(self.out_proj(out).view(B, self.K, d))


class SoftTemporalRuleScorer(nn.Module):
    """
    Learns soft temporal rules differentiably.

    Instead of mining rules symbolically (slow, brittle),
    we learn a neural approximation that captures the same
    temporal patterns but generalizes to unseen entities.

    Architecture:
    1. Encode subject's relation history as sequence
    2. Attend over history to find relevant patterns
    3. Predict which relations are likely next
    4. Score candidate (s, r, o, t) using predicted relation dist

    This approximates rules like:
    IF history contains (r1, r2) sequence THEN r3 is likely
    """
    def __init__(self, embedding_dim, num_relations, num_anchors,
                 max_history=20, num_heads=4):
        super().__init__()
        self.d   = embedding_dim
        self.R   = num_relations
        self.K   = num_anchors
        self.L   = max_history

        # Encode relation sequence
        self.rel_seq_emb  = nn.Linear(embedding_dim, embedding_dim)
        self.time_seq_emb = nn.Linear(1, embedding_dim)

        # Temporal pattern attention
        # Query: what relation are we trying to predict?
        # Key/Value: what patterns exist in history?
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj   = nn.Linear(embedding_dim * 2, embedding_dim)
        self.val_proj   = nn.Linear(embedding_dim * 2, embedding_dim)

        # Predict next relation distribution
        self.relation_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_relations)
        )

        # Rule confidence gate
        # How much to trust rules vs embedding score?
        self.confidence_gate = nn.Sequential(
            nn.Linear(embedding_dim + 1, 1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                query_rel_emb, rel_emb_table):
        """
        rel_ids:       (B, L) subject history relation ids
        timestamps:    (B, L) normalized timestamps
        lengths:       (B,)   valid history length
        tau:           (B,)   query timestamp
        query_rel_emb: (B, d) embedding of query relation r
        rel_emb_table: (R, d) all relation embeddings

        Returns:
            rule_score:  (B,)   soft rule confidence for query relation
            rule_probs:  (B, R) predicted relation distribution
            conf_weight: (B,)   how much to trust rules
        """
        B, L = rel_ids.shape
        dev  = rel_ids.device

        # Encode history relations and time deltas
        r_embs  = rel_emb_table[rel_ids]                    # (B, L, d)
        r_enc   = torch.relu(self.rel_seq_emb(r_embs))

        t_delta = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)
        t_enc   = torch.tanh(self.time_seq_emb(t_delta))

        # Combined event representation
        events  = torch.cat([r_enc, t_enc], dim=-1)         # (B, L, 2d)

        # Valid mask
        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()              # (B, L)

        # Attention: query relation attends over history
        Q = self.query_proj(query_rel_emb).unsqueeze(1)     # (B, 1, d)
        K = self.key_proj(events)                            # (B, L, d)
        V = self.val_proj(events)                            # (B, L, d)

        attn = (Q * K).sum(dim=-1) / (self.d ** 0.5)        # (B, L)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Pattern context
        pattern_ctx = (attn.unsqueeze(-1) * V).sum(dim=1)   # (B, d)
        pattern_ctx = self.norm(pattern_ctx)

        # Predict relation distribution
        combined   = torch.cat([pattern_ctx, query_rel_emb], dim=-1)
        rule_logits = self.relation_head(combined)            # (B, R)
        rule_probs  = F.softmax(rule_logits, dim=-1)          # (B, R)

        # Rule confidence: how much history do we have?
        hist_ratio   = (lengths.float() / self.L).clamp(0, 1).unsqueeze(-1)
        conf_input   = torch.cat([pattern_ctx, hist_ratio], dim=-1)
        conf_weight  = self.confidence_gate(conf_input).squeeze(-1)  # (B,)

        # Score for specific query relation
        # rule_probs[b, r] = P(next relation = r | history)
        # We need scores for all candidate objects
        # Use: score(o) = rule_probs[r] * neighbor_sim(pattern_ctx, o)
        rule_score = rule_probs.gather(
            1, torch.zeros(B, 1, dtype=torch.long, device=dev)
        ).squeeze(1)  # placeholder — computed per-relation in forward

        return pattern_ctx, rule_probs, conf_weight


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim,
                 num_anchors, sigma=1.0, dropout=0.2,
                 score_func='distmult', max_history=20,
                 rule_weight=0.3):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.sigma         = sigma
        self.max_history   = max_history
        self.rule_weight   = rule_weight

        self.anchor_times = nn.Parameter(torch.randn(num_anchors))

        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        # KARMA-T components
        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)

        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # KARMA base encoder
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-RL: soft rule scorer
        self.rule_scorer = SoftTemporalRuleScorer(
            embedding_dim, num_relations, num_anchors, max_history)

        # Combine embedding score + rule score
        self.score_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
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
        B, L = rel_ids.shape
        dev  = rel_ids.device
        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)
        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()
        has_history = (lengths > 0).float().unsqueeze(-1)
        anchor_reps = self.encoder(rel_embs, nbr_embs, time_deltas, mask)
        alpha       = self.anchor_attention(tau)
        encoded     = (alpha.unsqueeze(-1) * anchor_reps).sum(dim=1)
        cold        = (self.cold_start.unsqueeze(0) *
                       alpha.unsqueeze(-1)).sum(dim=1)
        return has_history * encoded + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        ind_emb = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau)
        ind_emb = self.layer_norm(self.dropout(ind_emb))
        if entity_ids is None:
            return ind_emb
        trans_emb  = self.layer_norm(self.dropout(
            self.entity_table(entity_ids)))
        hist_ratio = (lengths.float() / max(self.max_history,1)).clamp(0,1)
        gate_input = torch.cat([ind_emb, trans_emb,
                                 hist_ratio.unsqueeze(-1)], dim=-1)
        alpha_gate = self.gate_net(gate_input)
        return self.layer_norm(
            alpha_gate * ind_emb + (1.0 - alpha_gate) * trans_emb)

    def score_with_rules(self, s_emb, r_ids, all_obj_embs,
                          s_rels, s_times, s_lens, taus):
        """
        Compute combined embedding + rule scores.

        s_emb:       (B, d)  subject embedding
        r_ids:       (B,)    relation ids
        all_obj_embs:(N, d)  all entity embeddings
        Returns:     (B, N)  combined scores for all objects
        """
        B = s_emb.size(0)
        r_embs = self.relation_embeddings[r_ids]             # (B, d)

        # Embedding-based scores
        sr       = s_emb * r_embs                            # (B, d)
        emb_scores = sr @ all_obj_embs.t()                   # (B, N)

        # Rule-based scores
        pattern_ctx, rule_probs, conf_w = self.rule_scorer(
            s_rels, None, s_times, s_lens, taus,
            r_embs, self.relation_embeddings)                 # pattern_ctx:(B,d)

        # Rule scores: pattern_ctx similarity to all objects
        rule_scores = pattern_ctx @ all_obj_embs.t()         # (B, N)

        # Learned combination gate
        gate_input  = torch.stack([
            emb_scores.mean(dim=1),
            rule_scores.mean(dim=1)
        ], dim=-1)                                           # (B, 2)
        gate        = self.score_gate(gate_input)            # (B, 1)

        # Confident in rules when history is rich
        conf_gate   = (conf_w * self.rule_weight).unsqueeze(-1)  # (B, 1)

        combined = ((1 - conf_gate) * emb_scores +
                    conf_gate * rule_scores)                  # (B, N)
        return combined, rule_probs

    def rule_supervision_loss(self, r_ids, rule_probs):
        """
        Supervise rule scorer: predicted next relation should
        match actual query relation. This teaches the rule
        scorer to predict temporal patterns correctly.
        """
        return F.cross_entropy(rule_probs, r_ids)

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus, s_eids=None, o_eids=None):
        s_emb = self.entity_embed(s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_emb = self.entity_embed(o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])
        return (s_emb * r_emb * o_emb).sum(dim=-1)
