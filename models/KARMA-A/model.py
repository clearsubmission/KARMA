"""
KARMA-A: Knowledge Anchors with Adaptive Anchor Learning
         + Transductive-Inductive Hybrid Encoder

New in KARMA-A vs KARMA-T:
- Entity-type-specific anchor clusters
- Each entity learns WHICH cluster of anchors to use
- Different temporal patterns for different entity types
  e.g. political leaders have different temporal dynamics than countries
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
        events = torch.cat([r_enc, n_enc, t_enc], dim=-1)       # (B, L, 3d)

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
        return self.out_proj(out).view(B, self.K, d)             # (B, K, d)


class AdaptiveAnchorAttention(nn.Module):
    """
    KARMA-A core novelty: entity-type-specific anchor clusters.
    
    Instead of one shared set of K anchor timepoints,
    learn C clusters of anchors. Each entity learns which
    cluster best describes its temporal pattern.
    
    Political leaders -> cluster A (election cycles, ~4 year patterns)
    Countries         -> cluster B (slower, decade-scale patterns)
    Organizations     -> cluster C (medium-term patterns)
    
    The cluster assignment is learned from the entity's history context.
    """
    def __init__(self, embedding_dim, num_anchors, num_clusters=4, sigma=1.0):
        super().__init__()
        self.K = num_anchors
        self.C = num_clusters
        self.sigma = sigma

        # C clusters of K anchor timepoints each
        self.anchor_clusters = nn.Parameter(
            torch.randn(num_clusters, num_anchors))

        # Cluster assignment network: context -> cluster weights
        self.cluster_assign = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_clusters),
        )

        # Per-cluster sigma (different temporal scales)
        self.log_sigmas = nn.Parameter(torch.zeros(num_clusters))

    def forward(self, tau, context):
        """
        tau:     (B,)  query timestamp
        context: (B, d) entity context from encoder
        Returns: (B, K) adaptive anchor attention weights
        """
        B = tau.size(0)

        # Which cluster does each entity belong to?
        cluster_w = F.softmax(
            self.cluster_assign(context), dim=-1)               # (B, C)

        # Weighted combination of cluster anchor timepoints
        # cluster_w: (B, C), anchor_clusters: (C, K)
        entity_anchors = cluster_w @ self.anchor_clusters       # (B, K)

        # Per-cluster sigma (learn different temporal scales)
        sigmas = torch.exp(self.log_sigmas)                     # (C,)
        entity_sigma = (cluster_w * sigmas.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, 1)

        # Attention over entity-specific anchors
        diff  = tau.unsqueeze(1) - entity_anchors               # (B, K)
        log_w = -diff.pow(2) / (entity_sigma + 1e-8)
        return F.softmax(log_w, dim=-1), cluster_w              # (B, K), (B, C)


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult',
                 max_history=20, num_clusters=4):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.C             = num_clusters
        self.sigma         = sigma
        self.score_func    = score_func
        self.max_history   = max_history

        # Relation embeddings
        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        # Neighbor embeddings
        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        # Transductive entity table (KARMA-T component)
        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)

        # Inductive neighborhood encoder
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-A: Adaptive anchor attention
        self.adaptive_anchor = AdaptiveAnchorAttention(
            embedding_dim, num_anchors, num_clusters, sigma)

        # Gating: blend transductive + inductive
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # Cold start embedding
        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau):
        """
        Inductive embedding via neighborhood encoder + adaptive anchors.
        Core KARMA-A contribution: anchors adapt to entity type.
        """
        B, L = rel_ids.shape
        dev  = rel_ids.device

        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)

        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()

        has_history = (lengths > 0).float().unsqueeze(-1)

        # Encode neighborhood -> (B, K, d)
        anchor_reps = self.encoder(rel_embs, nbr_embs, time_deltas, mask)

        # Get context for adaptive anchor (mean of valid neighbor embs)
        denom   = mask.sum(dim=1, keepdim=True).clamp(min=1)
        context = (nbr_embs * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, d)

        # KARMA-A: adaptive anchor attention based on entity context
        alpha, cluster_w = self.adaptive_anchor(tau, context)   # (B, K), (B, C)

        # Weighted sum over anchors -> (B, d)
        encoded = (alpha.unsqueeze(-1) * anchor_reps).sum(dim=1)

        # Cold start: also use adaptive anchors
        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        out = has_history * encoded + (1.0 - has_history) * cold
        return self.layer_norm(self.dropout(out)), cluster_w

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        """
        KARMA-A full hybrid embedding:
        1. Inductive encoder with adaptive anchors (KARMA-A)
        2. Transductive lookup (KARMA-T)
        3. Learned gating between the two
        """
        ind_emb, cluster_w = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau)

        if entity_ids is None:
            return ind_emb, cluster_w

        # Transductive lookup
        trans_emb = self.layer_norm(
            self.dropout(self.entity_table(entity_ids)))

        # Learned gate
        hist_ratio = (lengths.float() / max(self.max_history, 1)).clamp(0, 1)
        gate_input = torch.cat([
            ind_emb, trans_emb,
            hist_ratio.unsqueeze(-1)
        ], dim=-1)
        alpha_gate = self.gate_net(gate_input)                   # (B, 1)

        hybrid = alpha_gate * ind_emb + (1.0 - alpha_gate) * trans_emb
        return self.layer_norm(hybrid), cluster_w

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus,
                s_eids=None, o_eids=None):

        s_emb, s_cw = self.entity_embed(
            s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_emb, o_cw = self.entity_embed(
            o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])

        pos_scores = (s_emb * r_emb * o_emb).sum(dim=-1)
        return pos_scores, s_cw, o_cw
