"""
KARMA-R: Knowledge Anchors with Relational Path Augmentation

New in KARMA-R vs KARMA-T:
- Multi-hop relational paths as additional context
- 1-hop: direct neighbors (same as KARMA-T)
- 2-hop: neighbors of neighbors
- Path encoder: encodes relation sequences, not just single relations
- Richer structural context → better entity representations
- Especially helps for entities with sparse direct history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
    """1-hop encoder — same as KARMA-T."""
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


class RelationalPathEncoder(nn.Module):
    """
    KARMA-R core: encodes 2-hop relational paths.
    Uses attention-weighted mean pooling (stable, no MultiheadAttention).
    """
    def __init__(self, embedding_dim, num_anchors, max_paths=10):
        super().__init__()
        self.d         = embedding_dim
        self.K         = num_anchors
        self.max_paths = max_paths

        self.r1_proj   = nn.Linear(embedding_dim, embedding_dim)
        self.r2_proj   = nn.Linear(embedding_dim, embedding_dim)
        self.n2_proj   = nn.Linear(embedding_dim, embedding_dim)
        self.t1_proj   = nn.Linear(1, embedding_dim)
        self.t2_proj   = nn.Linear(1, embedding_dim)

        # Path importance scoring
        self.path_score = nn.Linear(embedding_dim, 1)
        self.out_proj   = nn.Linear(embedding_dim, num_anchors * embedding_dim)
        self.norm       = nn.LayerNorm(embedding_dim)

    def forward(self, r1_embs, r2_embs, n2_embs, t1_deltas, t2_deltas, mask):
        B, P, d = r1_embs.shape

        r1 = torch.relu(self.r1_proj(r1_embs))
        r2 = torch.relu(self.r2_proj(r2_embs))
        n2 = torch.relu(self.n2_proj(n2_embs))
        t1 = torch.tanh(self.t1_proj(t1_deltas))
        t2 = torch.tanh(self.t2_proj(t2_deltas))

        path_repr = r1 + r2 + n2 + t1 + t2                      # (B, P, d)

        # Attention-weighted pooling
        scores = self.path_score(path_repr).squeeze(-1)          # (B, P)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)             # all-masked rows

        pooled = (weights.unsqueeze(-1) * path_repr).sum(dim=1)  # (B, d)
        pooled = self.norm(pooled)

        return self.out_proj(pooled).view(B, self.K, d)          # (B, K, d)


class KARMA(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult',
                 max_history=20, max_paths=10, path_weight=0.5):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.sigma         = sigma
        self.score_func    = score_func
        self.max_history   = max_history
        self.max_paths     = max_paths
        self.path_weight   = path_weight

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

        # 1-hop encoder (same as KARMA-T)
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-R: 2-hop path encoder
        self.path_encoder = RelationalPathEncoder(
            embedding_dim, num_anchors, max_paths)

        # Learned weight for combining 1-hop and 2-hop representations
        self.hop_gate = nn.Sequential(
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

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                        path_r1=None, path_r2=None, path_n2=None,
                        path_t1=None, path_t2=None, path_mask=None):
        """
        Inductive embedding with optional 2-hop path augmentation.

        1-hop inputs (required):
            rel_ids, nbr_ids, timestamps, lengths, tau

        2-hop path inputs (optional — None = use 1-hop only):
            path_r1:   (B, P) relation ids for first hop
            path_r2:   (B, P) relation ids for second hop
            path_n2:   (B, P) entity ids for second-hop neighbor
            path_t1:   (B, P) timestamps for first hop
            path_t2:   (B, P) timestamps for second hop
            path_mask: (B, P) valid path mask
        """
        B, L = rel_ids.shape
        dev  = rel_ids.device

        # === 1-hop encoding ===
        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)

        mask_1hop = (torch.arange(L, device=dev).unsqueeze(0)
                     < lengths.unsqueeze(1)).float()
        has_history = (lengths > 0).float().unsqueeze(-1)

        anchor_reps_1hop = self.encoder(
            rel_embs, nbr_embs, time_deltas, mask_1hop)          # (B, K, d)

        # === 2-hop encoding (if paths provided) ===
        if path_r1 is not None and path_mask is not None and path_mask.sum() > 0:
            r1_embs = self.relation_embeddings[path_r1]           # (B, P, d)
            r2_embs = self.relation_embeddings[path_r2]           # (B, P, d)
            n2_embs = self.neighbor_embeddings[path_n2]           # (B, P, d)
            t1_deltas = (tau.unsqueeze(1) - path_t1).unsqueeze(-1)
            t2_deltas = (tau.unsqueeze(1) - path_t2).unsqueeze(-1)

            anchor_reps_2hop = self.path_encoder(
                r1_embs, r2_embs, n2_embs,
                t1_deltas, t2_deltas, path_mask)                  # (B, K, d)

            # Learned combination of 1-hop and 2-hop
            alpha   = self.anchor_attention(tau)                   # (B, K)
            emb_1   = (alpha.unsqueeze(-1) * anchor_reps_1hop).sum(dim=1)
            emb_2   = (alpha.unsqueeze(-1) * anchor_reps_2hop).sum(dim=1)

            gate    = self.hop_gate(torch.cat([emb_1, emb_2], dim=-1))
            encoded = gate * emb_1 + (1.0 - gate) * emb_2
        else:
            # Fallback to 1-hop only
            alpha   = self.anchor_attention(tau)
            encoded = (alpha.unsqueeze(-1) * anchor_reps_1hop).sum(dim=1)

        cold    = (self.cold_start.unsqueeze(0) *
                   alpha.unsqueeze(-1)).sum(dim=1)

        return has_history * encoded + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None,
                     path_r1=None, path_r2=None, path_n2=None,
                     path_t1=None, path_t2=None, path_mask=None):
        """KARMA-R hybrid embedding with optional path augmentation."""
        ind_emb = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau,
            path_r1, path_r2, path_n2, path_t1, path_t2, path_mask)
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

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus, s_eids=None, o_eids=None,
                s_path_r1=None, s_path_r2=None, s_path_n2=None,
                s_path_t1=None, s_path_t2=None, s_path_mask=None):

        s_emb = self.entity_embed(
            s_rids, s_nids, s_ts, s_lens, taus, s_eids,
            s_path_r1, s_path_r2, s_path_n2,
            s_path_t1, s_path_t2, s_path_mask)
        o_emb = self.entity_embed(
            o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        r_emb = self.dropout(self.relation_embeddings[r_ids])

        pos_scores = (s_emb * r_emb * o_emb).sum(dim=-1)
        return pos_scores
