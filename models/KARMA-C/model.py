"""
KARMA-C: Knowledge Anchors with Contrastive Temporal Pretraining

New in KARMA-C vs KARMA-T:
- Contrastive learning over temporal entity representations
- Same entity at nearby timestamps → similar embeddings (positive pairs)
- Different entities at same timestamp → dissimilar embeddings (negative pairs)
- Temporal smoothness: entity embeddings should change gradually over time
- Better generalization to unseen entities via improved encoder pretraining
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

        out = (attn.unsqueeze(-1) * V).sum(dim=1)
        out = self.norm(out)
        return self.out_proj(out).view(B, self.K, d)


class TemporalContrastiveHead(nn.Module):
    """
    KARMA-C core: projects entity embeddings into contrastive space.
    Separate projection head prevents contrastive loss from
    interfering with link prediction objective.
    """
    def __init__(self, embedding_dim, projection_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult',
                 max_history=20, temperature=0.07, projection_dim=128):
        super().__init__()
        self.num_entities   = num_entities
        self.num_relations  = num_relations
        self.d              = embedding_dim
        self.K              = num_anchors
        self.sigma          = sigma
        self.score_func     = score_func
        self.max_history    = max_history
        self.temperature    = temperature

        # Core KARMA: anchor timepoints
        self.anchor_times = nn.Parameter(torch.randn(num_anchors))

        # Relation embeddings
        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        # Neighbor embeddings
        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        # Transductive entity table (from KARMA-T)
        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)

        # Learned gating (from KARMA-T)
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # Inductive encoder
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-C: contrastive projection head
        self.contrastive_head = TemporalContrastiveHead(
            embedding_dim, projection_dim)

        # Cold start
        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def anchor_attention(self, tau):
        """tau: (B,) -> (B, K)"""
        diff  = tau.unsqueeze(1) - self.anchor_times.unsqueeze(0)
        return F.softmax(-diff.pow(2) / (self.sigma + 1e-8), dim=-1)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau):
        """Pure inductive embedding via neighborhood encoder."""
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

        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        return has_history * encoded + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        """KARMA-C hybrid embedding (same as KARMA-T)."""
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

    def contrastive_loss(self, anchor_emb, positive_emb, negative_embs,
                         temperature=None):
        """
        InfoNCE contrastive loss.

        anchor_emb:    (B, d)   entity embedding at time t
        positive_emb:  (B, d)   same entity at nearby time t' (positive pair)
        negative_embs: (B, N, d) different entities at time t (negatives)

        Objective: anchor should be close to positive, far from negatives
        """
        if temperature is None:
            temperature = self.temperature

        # Project to contrastive space
        z_a = self.contrastive_head(anchor_emb)    # (B, proj_dim)
        z_p = self.contrastive_head(positive_emb)  # (B, proj_dim)
        z_n = self.contrastive_head(
            negative_embs.view(-1, self.d)).view(
            negative_embs.size(0), negative_embs.size(1), -1)  # (B, N, proj_dim)

        # Positive similarity
        pos_sim = (z_a * z_p).sum(dim=-1) / temperature        # (B,)

        # Negative similarities
        neg_sim = torch.bmm(
            z_n, z_a.unsqueeze(-1)).squeeze(-1) / temperature  # (B, N)

        # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, N+1)
        labels = torch.zeros(logits.size(0), dtype=torch.long,
                             device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def temporal_smoothness_loss(self, emb_t1, emb_t2, time_diff):
        """
        Temporal smoothness: entity embeddings at nearby times
        should be similar. Distance proportional to time gap.

        emb_t1:    (B, d)
        emb_t2:    (B, d)
        time_diff: (B,) normalized time difference
        """
        # Cosine distance between consecutive embeddings
        cos_sim   = F.cosine_similarity(emb_t1, emb_t2, dim=-1)  # (B,)
        cos_dist  = 1.0 - cos_sim                                  # (B,)

        # Weight by inverse time difference — closer times should be more similar
        weight    = torch.exp(-time_diff.abs() * 5.0)              # (B,)
        loss      = (weight * cos_dist).mean()
        return loss

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus,
                s_eids=None, o_eids=None):

        s_emb = self.entity_embed(s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_emb = self.entity_embed(o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])

        pos_scores = (s_emb * r_emb * o_emb).sum(dim=-1)
        return pos_scores
