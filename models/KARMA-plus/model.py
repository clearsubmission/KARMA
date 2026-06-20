"""
KARMA+: Full Stack — All Components Combined

Stacks:
- Anchor attention (KARMA-base)
- Transductive hybrid gate (KARMA-T)
- Adaptive anchor clustering (KARMA-A)
- Future-aware encoder (KARMA-F)
- Contrastive projection head (KARMA-C)
- 2-hop relational paths (KARMA-R)

Design principle: each component is gated/weighted so it can be
turned off if it hurts for a specific dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
    """1-hop history encoder."""
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


class RelationalPathEncoder(nn.Module):
    """2-hop path encoder (KARMA-R)."""
    def __init__(self, embedding_dim, num_anchors, max_paths=10):
        super().__init__()
        self.d = embedding_dim
        self.K = num_anchors
        self.r1_proj    = nn.Linear(embedding_dim, embedding_dim)
        self.r2_proj    = nn.Linear(embedding_dim, embedding_dim)
        self.n2_proj    = nn.Linear(embedding_dim, embedding_dim)
        self.t1_proj    = nn.Linear(1, embedding_dim)
        self.t2_proj    = nn.Linear(1, embedding_dim)
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
        path_repr = r1 + r2 + n2 + t1 + t2
        scores  = self.path_score(path_repr).squeeze(-1)
        scores  = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled  = (weights.unsqueeze(-1) * path_repr).sum(dim=1)
        return self.norm(self.out_proj(pooled).view(B, self.K, d))


class AdaptiveAnchorAttention(nn.Module):
    """Entity-type-specific anchor clusters (KARMA-A)."""
    def __init__(self, embedding_dim, num_anchors, num_clusters=4, sigma=1.0):
        super().__init__()
        self.K = num_anchors
        self.C = num_clusters
        self.sigma = sigma
        self.anchor_clusters = nn.Parameter(
            torch.randn(num_clusters, num_anchors))
        self.cluster_assign = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_clusters),
        )
        self.log_sigmas = nn.Parameter(torch.zeros(num_clusters))

    def forward(self, tau, context):
        cluster_w    = F.softmax(self.cluster_assign(context), dim=-1)
        entity_anchors = cluster_w @ self.anchor_clusters
        sigmas       = torch.exp(self.log_sigmas)
        entity_sigma = (cluster_w * sigmas.unsqueeze(0)).sum(dim=-1, keepdim=True)
        diff  = tau.unsqueeze(1) - entity_anchors
        log_w = -diff.pow(2) / (entity_sigma + 1e-8)
        return F.softmax(log_w, dim=-1), cluster_w


class FutureContextPredictor(nn.Module):
    """Future-aware encoder (KARMA-F)."""
    def __init__(self, embedding_dim, num_relations, num_anchors):
        super().__init__()
        self.relation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_relations),
        )
        self.future_encoder = nn.Sequential(
            nn.Linear(embedding_dim + num_relations, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_anchors * embedding_dim)
        )

    def forward(self, past_emb, num_anchors, d):
        B = past_emb.size(0)
        rel_probs   = F.softmax(self.relation_predictor(past_emb), dim=-1)
        future_ctx  = torch.cat([past_emb, rel_probs], dim=-1)
        future_repr = self.future_encoder(future_ctx)
        return future_repr.view(B, num_anchors, d), rel_probs


class ContrastiveHead(nn.Module):
    """Contrastive projection head (KARMA-C)."""
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
                 sigma=1.0, dropout=0.2, score_func='distmult', max_history=20,
                 num_clusters=4, temperature=0.07, projection_dim=128,
                 max_paths=10):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.C             = num_clusters
        self.sigma         = sigma
        self.score_func    = score_func
        self.max_history   = max_history
        self.temperature   = temperature
        self.max_paths     = max_paths

        # Shared embeddings
        self.anchor_times = nn.Parameter(torch.randn(num_anchors))
        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)
        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        # KARMA-T: transductive lookup + gate
        self.entity_table = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_table.weight)
        self.gate_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

        # KARMA-base: 1-hop encoder
        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        # KARMA-R: 2-hop path encoder
        self.path_encoder = RelationalPathEncoder(
            embedding_dim, num_anchors, max_paths)
        self.hop_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid()
        )

        # KARMA-A: adaptive anchor clusters
        self.adaptive_anchor = AdaptiveAnchorAttention(
            embedding_dim, num_anchors, num_clusters, sigma)

        # KARMA-F: future context predictor
        self.future_predictor = FutureContextPredictor(
            embedding_dim, num_relations, num_anchors)
        self.future_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid()
        )

        # KARMA-C: contrastive head
        self.contrastive_head = ContrastiveHead(embedding_dim, projection_dim)

        # Component weighting gates (learn which components matter per dataset)
        self.component_gate = nn.Sequential(
            nn.Linear(embedding_dim, 3),  # 3 components: 1hop, path, future
            nn.Softmax(dim=-1)
        )

        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def anchor_attention(self, tau):
        """Fixed anchor attention (KARMA-base)."""
        diff  = tau.unsqueeze(1) - self.anchor_times.unsqueeze(0)
        return F.softmax(-diff.pow(2) / (self.sigma + 1e-8), dim=-1)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                        path_r1=None, path_r2=None, path_n2=None,
                        path_t1=None, path_t2=None, path_mask=None):
        """Full inductive embedding: 1-hop + 2-hop + adaptive anchors + future."""
        B, L = rel_ids.shape
        dev  = rel_ids.device

        # === 1-hop encoding ===
        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)
        mask_1hop   = (torch.arange(L, device=dev).unsqueeze(0)
                       < lengths.unsqueeze(1)).float()
        has_history = (lengths > 0).float().unsqueeze(-1)

        anchor_reps_1hop = self.encoder(
            rel_embs, nbr_embs, time_deltas, mask_1hop)        # (B, K, d)

        # === Context for adaptive anchors ===
        denom   = mask_1hop.sum(dim=1, keepdim=True).clamp(min=1)
        context = (nbr_embs * mask_1hop.unsqueeze(-1)).sum(dim=1) / denom

        # === KARMA-A: adaptive anchor attention ===
        alpha, cluster_w = self.adaptive_anchor(tau, context)   # (B, K), (B, C)

        # === 1-hop embedding ===
        emb_1hop = (alpha.unsqueeze(-1) * anchor_reps_1hop).sum(dim=1)  # (B, d)

        # === KARMA-R: 2-hop paths ===
        if path_r1 is not None and path_mask is not None:
            r1e = self.relation_embeddings[path_r1]
            r2e = self.relation_embeddings[path_r2]
            n2e = self.neighbor_embeddings[path_n2]
            t1d = (tau.unsqueeze(1) - path_t1).unsqueeze(-1)
            t2d = (tau.unsqueeze(1) - path_t2).unsqueeze(-1)
            anchor_reps_2hop = self.path_encoder(
                r1e, r2e, n2e, t1d, t2d, path_mask)
            emb_2hop = (alpha.unsqueeze(-1) * anchor_reps_2hop).sum(dim=1)
            hop_gate = self.hop_gate(
                torch.cat([emb_1hop, emb_2hop], dim=-1))
            emb_structural = hop_gate * emb_1hop + (1-hop_gate) * emb_2hop
        else:
            emb_structural = emb_1hop

        # === KARMA-F: future context ===
        anchor_reps_future, rel_probs = self.future_predictor(
            emb_structural, self.K, self.d)
        emb_future = (alpha.unsqueeze(-1) * anchor_reps_future).sum(dim=1)
        future_gate = self.future_gate(
            torch.cat([emb_structural, emb_future], dim=-1))
        emb_combined = (future_gate * emb_structural +
                        (1-future_gate) * emb_future)

        # === Cold start fallback ===
        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        out = has_history * emb_combined + (1.0 - has_history) * cold
        return out, cluster_w, rel_probs

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None,
                     path_r1=None, path_r2=None, path_n2=None,
                     path_t1=None, path_t2=None, path_mask=None):
        """KARMA+: full hybrid embedding."""
        ind_emb, cluster_w, rel_probs = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau,
            path_r1, path_r2, path_n2, path_t1, path_t2, path_mask)
        ind_emb = self.layer_norm(self.dropout(ind_emb))

        if entity_ids is None:
            return ind_emb, cluster_w, rel_probs

        # KARMA-T: transductive hybrid gate
        trans_emb = self.layer_norm(
            self.dropout(self.entity_table(entity_ids)))
        hist_ratio = (lengths.float() / max(self.max_history, 1)).clamp(0, 1)
        gate_input = torch.cat([
            ind_emb, trans_emb, hist_ratio.unsqueeze(-1)], dim=-1)
        alpha_gate = self.gate_net(gate_input)
        hybrid = alpha_gate * ind_emb + (1.0 - alpha_gate) * trans_emb
        return self.layer_norm(hybrid), cluster_w, rel_probs

    def contrastive_loss(self, anchor_emb, positive_emb, negative_embs):
        """InfoNCE loss (KARMA-C)."""
        z_a = self.contrastive_head(anchor_emb)
        z_p = self.contrastive_head(positive_emb)
        z_n = self.contrastive_head(
            negative_embs.view(-1, self.d)).view(
            negative_embs.size(0), negative_embs.size(1), -1)
        pos_sim = (z_a * z_p).sum(dim=-1) / self.temperature
        neg_sim = torch.bmm(z_n, z_a.unsqueeze(-1)).squeeze(-1) / self.temperature
        logits  = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels  = torch.zeros(logits.size(0), dtype=torch.long,
                              device=logits.device)
        return F.cross_entropy(logits, labels)

    def cluster_diversity_loss(self, cluster_w):
        """Prevent cluster collapse (KARMA-A)."""
        mean_usage = cluster_w.mean(dim=0)
        return (mean_usage * (mean_usage + 1e-8).log()).sum()

    def anchor_spread_loss(self):
        """Keep anchors spread across time."""
        loss = torch.tensor(0.0, device=self.anchor_times.device)
        anchors = self.adaptive_anchor.anchor_clusters
        for c in range(self.C):
            if anchors[c].size(0) > 1:
                spread = torch.pdist(anchors[c].unsqueeze(1), p=2).mean()
                loss   = loss + torch.clamp(1.0 - spread, min=0)
        return loss

    def future_relation_loss(self, entity_ids, rel_ids):
        """Auxiliary future relation prediction (KARMA-F)."""
        past_emb   = self.layer_norm(self.entity_table(entity_ids))
        rel_logits = self.future_predictor.relation_predictor(past_emb)
        return F.cross_entropy(rel_logits, rel_ids)

    def forward(self, s_rids, s_nids, s_ts, s_lens,
                o_rids, o_nids, o_ts, o_lens,
                r_ids, taus, s_eids=None, o_eids=None):
        s_result = self.entity_embed(
            s_rids, s_nids, s_ts, s_lens, taus, s_eids)
        o_result = self.entity_embed(
            o_rids, o_nids, o_ts, o_lens, taus, o_eids)
        s_emb = s_result[0]
        o_emb = o_result[0]
        r_emb = self.dropout(self.relation_embeddings[r_ids])
        return (s_emb * r_emb * o_emb).sum(dim=-1)
