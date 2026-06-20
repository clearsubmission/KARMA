import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def load_quadruples(file_path):
    quads = []
    if not os.path.exists(file_path):
        return np.array([], dtype=np.int64).reshape(0, 4)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    quads.append([int(p) for p in parts[:4]])
                except ValueError:
                    continue
    return np.array(quads, dtype=np.int64)


def build_history_index(quads):
    """Build sorted history index."""
    num_ent = int(quads[:, [0,2]].max()) + 1
    ents = np.concatenate([quads[:,0], quads[:,2]])
    ts   = np.concatenate([quads[:,3], quads[:,3]])
    rs   = np.concatenate([quads[:,1], quads[:,1]])
    nbs  = np.concatenate([quads[:,2], quads[:,0]])

    order = np.lexsort((ts, ents))
    ents  = ents[order]; ts = ts[order]
    rs    = rs[order];   nbs = nbs[order]

    ent_start = np.zeros(num_ent + 2, dtype=np.int64)
    np.add.at(ent_start[1:], ents, 1)
    np.cumsum(ent_start, out=ent_start)
    return ts, rs, nbs, ent_start, num_ent


def precompute_histories_fast(quads, ts, rs, nbs, ent_start,
                               t_min, t_range, max_history=20):
    N = len(quads); L = max_history
    s_rels  = np.zeros((N, L), dtype=np.int64)
    s_nbrs  = np.zeros((N, L), dtype=np.int64)
    s_times = np.zeros((N, L), dtype=np.float32)
    s_lens  = np.zeros(N,      dtype=np.int64)
    o_rels  = np.zeros((N, L), dtype=np.int64)
    o_nbrs  = np.zeros((N, L), dtype=np.int64)
    o_times = np.zeros((N, L), dtype=np.float32)
    o_lens  = np.zeros(N,      dtype=np.int64)

    for i in range(N):
        s, r, o, t = int(quads[i,0]), int(quads[i,1]), int(quads[i,2]), int(quads[i,3])
        for eid, rr, nn, lt, lnb, ll in [
            (s, s_rels, s_nbrs, s_times, None, s_lens),
            (o, o_rels, o_nbrs, o_times, None, o_lens)
        ]:
            lo, hi = (int(ent_start[eid]), int(ent_start[eid+1]) if eid+1 < len(ent_start) else 0) if eid < len(ent_start) else (0, 0)
            if not isinstance(lo, int):
                lo, hi = lo, hi
            if hi > lo:
                ent_ts = ts[lo:hi]
                valid  = np.searchsorted(ent_ts, t, side='left')
                if valid > 0:
                    start = max(0, valid - L)
                    n     = valid - start
                    rr[i, :n]  = rs[lo+start:lo+valid]
                    nn[i, :n]  = nbs[lo+start:lo+valid]
                    lt[i, :n]  = (ts[lo+start:lo+valid].astype(np.float32) - t_min) / t_range
                    ll[i]      = n

    return s_rels, s_nbrs, s_times, s_lens, o_rels, o_nbrs, o_times, o_lens


def precompute_2hop_paths(quads, ts, rs, nbs, ent_start,
                           t_min, t_range, max_paths=10):
    """
    For each quad, find 2-hop paths through subject's neighbors.
    Path: s -[r1]-> n1 -[r2]-> n2 (both before query time t)
    """
    N = len(quads); P = max_paths

    s_pr1   = np.zeros((N, P), dtype=np.int64)   # first-hop relation
    s_pr2   = np.zeros((N, P), dtype=np.int64)   # second-hop relation
    s_pn2   = np.zeros((N, P), dtype=np.int64)   # second-hop neighbor
    s_pt1   = np.zeros((N, P), dtype=np.float32) # first-hop timestamp
    s_pt2   = np.zeros((N, P), dtype=np.float32) # second-hop timestamp
    s_pmask = np.zeros((N, P), dtype=np.float32) # valid path mask

    for i in range(N):
        s, r, o, t = int(quads[i,0]), int(quads[i,1]), int(quads[i,2]), int(quads[i,3])

        # Get s's 1-hop neighbors before t
        lo, hi = (int(ent_start[s]), int(ent_start[s+1]) if s+1 < len(ent_start) else 0) if s < len(ent_start) else (0, 0)
        if hi <= lo:
            continue

        ent_ts = ts[lo:hi]
        valid  = np.searchsorted(ent_ts, t, side='left')
        if valid == 0:
            continue

        # Sample up to sqrt(P) 1-hop neighbors
        n1_count = min(valid, int(P**0.5) + 1)
        n1_idxs  = np.arange(max(0, valid-n1_count), valid)

        path_count = 0
        for idx in n1_idxs:
            n1  = int(nbs[lo + idx])
            r1  = int(rs[lo + idx])
            t1  = float(ts[lo + idx])

            # Get n1's neighbors before t
            if n1+1 >= len(ent_start):
                continue
            lo2, hi2 = (int(ent_start[n1]), int(ent_start[n1+1]) if n1+1 < len(ent_start) else 0) if n1 < len(ent_start) else (0, 0)
            if hi2 <= lo2:
                continue

            n1_ts  = ts[lo2:hi2]
            valid2 = np.searchsorted(n1_ts, t, side='left')
            if valid2 == 0:
                continue

            # Take most recent 2-hop neighbors
            n2_count = min(valid2, max(1, P // n1_count))
            for idx2 in range(max(0, valid2-n2_count), valid2):
                if path_count >= P:
                    break
                s_pr1[i, path_count]   = r1
                s_pr2[i, path_count]   = int(rs[lo2 + idx2])
                s_pn2[i, path_count]   = int(nbs[lo2 + idx2])
                s_pt1[i, path_count]   = (t1 - t_min) / t_range
                s_pt2[i, path_count]   = (float(ts[lo2+idx2]) - t_min) / t_range
                s_pmask[i, path_count] = 1.0
                path_count += 1

            if path_count >= P:
                break

    return s_pr1, s_pr2, s_pn2, s_pt1, s_pt2, s_pmask


class TKGDataset:
    def __init__(self, data_dir, max_history=20, max_paths=10):
        self.data_dir    = data_dir
        self.max_history = max_history
        self.max_paths   = max_paths

        self.train = load_quadruples(os.path.join(data_dir, 'train.txt'))
        self.valid = load_quadruples(os.path.join(data_dir, 'valid.txt'))
        self.test  = load_quadruples(os.path.join(data_dir, 'test.txt'))

        if len(self.valid) == 0:
            n = len(self.train); cut = int(n*0.9)
            idx = np.random.permutation(n)
            self.valid = self.train[idx[cut:]]
            self.train = self.train[idx[:cut]]
            print(f"  No valid.txt — split: train={len(self.train)}, valid={len(self.valid)}")

        all_quads = np.concatenate([self.train, self.valid, self.test])
        self.num_entities  = int(all_quads[:,[0,2]].max()) + 1
        self.num_relations = int(all_quads[:,1].max()) + 1

        all_times    = all_quads[:,3].astype(np.float32)
        self.t_min   = float(all_times.min())
        self.t_max   = float(all_times.max())
        self.t_range = max(self.t_max - self.t_min, 1.0)

        train_ents           = set(self.train[:,0].tolist() + self.train[:,2].tolist())
        test_ents            = set(self.test[:,0].tolist()  + self.test[:,2].tolist())
        self.unseen_entities = test_ents - train_ents
        self.seen_entities   = test_ents & train_ents

        print(f"  Entities: {self.num_entities}, Relations: {self.num_relations}")
        print(f"  Timestamps: {self.t_min:.0f} to {self.t_max:.0f}")
        print(f"  Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}")
        print(f"  Unseen: {len(self.unseen_entities)}, Seen: {len(self.seen_entities)}")

        print("  Building history index...")
        self.ts, self.rs, self.nbs, self.ent_start, _ = build_history_index(self.train)

        print("  Precomputing 1-hop histories...")
        self.train_hist = precompute_histories_fast(
            self.train, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)
        self.valid_hist = precompute_histories_fast(
            self.valid, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)
        self.test_hist  = precompute_histories_fast(
            self.test, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)

        print("  Precomputing 2-hop paths...")
        self.train_paths = precompute_2hop_paths(
            self.train, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_paths)
        self.valid_paths = precompute_2hop_paths(
            self.valid, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_paths)
        self.test_paths  = precompute_2hop_paths(
            self.test, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_paths)
        print("  Done.")


class QuadrupleDataset(Dataset):
    def __init__(self, quads, t_min, t_range, hist, paths):
        self.quads = quads
        self.taus  = (quads[:,3].astype(np.float32) - t_min) / t_range
        (self.s_rels, self.s_nbrs, self.s_times, self.s_lens,
         self.o_rels, self.o_nbrs, self.o_times, self.o_lens) = hist
        (self.s_pr1, self.s_pr2, self.s_pn2,
         self.s_pt1, self.s_pt2, self.s_pmask) = paths

    def __len__(self): return len(self.quads)

    def __getitem__(self, idx):
        s, r, o, _ = self.quads[idx]
        return (int(s), int(r), int(o), float(self.taus[idx]),
                self.s_rels[idx], self.s_nbrs[idx],
                self.s_times[idx], int(self.s_lens[idx]),
                self.o_rels[idx], self.o_nbrs[idx],
                self.o_times[idx], int(self.o_lens[idx]),
                self.s_pr1[idx], self.s_pr2[idx], self.s_pn2[idx],
                self.s_pt1[idx], self.s_pt2[idx], self.s_pmask[idx])


def collate_fn(batch):
    (s_ids, r_ids, o_ids, taus,
     s_rels, s_nbrs, s_times, s_lens,
     o_rels, o_nbrs, o_times, o_lens,
     s_pr1, s_pr2, s_pn2, s_pt1, s_pt2, s_pmask) = zip(*batch)
    return (
        torch.tensor(s_ids,  dtype=torch.long),
        torch.tensor(r_ids,  dtype=torch.long),
        torch.tensor(o_ids,  dtype=torch.long),
        torch.tensor(taus,   dtype=torch.float),
        torch.tensor(np.stack(s_rels),  dtype=torch.long),
        torch.tensor(np.stack(s_nbrs),  dtype=torch.long),
        torch.tensor(np.stack(s_times), dtype=torch.float),
        torch.tensor(s_lens, dtype=torch.long),
        torch.tensor(np.stack(o_rels),  dtype=torch.long),
        torch.tensor(np.stack(o_nbrs),  dtype=torch.long),
        torch.tensor(np.stack(o_times), dtype=torch.float),
        torch.tensor(o_lens, dtype=torch.long),
        torch.tensor(np.stack(s_pr1),   dtype=torch.long),
        torch.tensor(np.stack(s_pr2),   dtype=torch.long),
        torch.tensor(np.stack(s_pn2),   dtype=torch.long),
        torch.tensor(np.stack(s_pt1),   dtype=torch.float),
        torch.tensor(np.stack(s_pt2),   dtype=torch.float),
        torch.tensor(np.stack(s_pmask), dtype=torch.float),
    )


def get_dataloaders(tkgdata, batch_size=512, num_workers=0):
    def mk(quads, hist, paths):
        return QuadrupleDataset(quads, tkgdata.t_min, tkgdata.t_range,
                                hist, paths)
    lkw = dict(batch_size=batch_size, num_workers=num_workers,
               collate_fn=collate_fn, pin_memory=False)
    return (DataLoader(mk(tkgdata.train, tkgdata.train_hist, tkgdata.train_paths),
                       shuffle=True,  **lkw),
            DataLoader(mk(tkgdata.valid, tkgdata.valid_hist, tkgdata.valid_paths),
                       shuffle=False, **lkw),
            DataLoader(mk(tkgdata.test,  tkgdata.test_hist,  tkgdata.test_paths),
                       shuffle=False, **lkw))


def build_filter_dict_normalized(all_quads, t_min, t_range):
    from collections import defaultdict
    filt = defaultdict(set)
    for s, r, o, t in all_quads:
        tau = round((float(t) - t_min) / t_range, 6)
        filt[(int(s), int(r), tau)].add(int(o))
    return filt
