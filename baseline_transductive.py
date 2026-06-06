import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from tqdm import tqdm


def load_quadruples(path):
    quads = []
    if not os.path.exists(path):
        return np.array([], dtype=np.int64).reshape(0,4)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    quads.append([int(p) for p in parts[:4]])
                except ValueError:
                    continue
    return np.array(quads, dtype=np.int64)


class TransTKG(nn.Module):
    def __init__(self, num_entities, num_relations, num_times, dim=200):
        super().__init__()
        self.ent_emb  = nn.Embedding(num_entities,  dim)
        self.rel_emb  = nn.Embedding(num_relations, dim)
        self.time_emb = nn.Embedding(num_times,     dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.time_emb.weight)

    def score_all(self, s, r, t):
        sr = self.ent_emb(s) + self.rel_emb(r) + self.time_emb(t)
        all_o = self.ent_emb.weight
        return -torch.cdist(sr.unsqueeze(1), all_o.unsqueeze(0)).squeeze(1)


class QuadDataset(Dataset):
    def __init__(self, quads):
        self.quads = quads
    def __len__(self): return len(self.quads)
    def __getitem__(self, i):
        s,r,o,t = self.quads[i]
        return int(s), int(r), int(o), int(t)


def build_filter(quads):
    filt = defaultdict(set)
    for s,r,o,t in quads:
        filt[(int(s),int(r),int(t))].add(int(o))
    return filt


@torch.no_grad()
def evaluate(model, quads, filter_dict, device, unseen_entities=None):
    model.eval()
    ranks, ranks_seen, ranks_unseen = [], [], []
    loader = DataLoader(QuadDataset(quads), batch_size=256, shuffle=False)
    for batch in tqdm(loader, leave=False):
        s,r,o,t = [x.to(device) for x in batch]
        scores = model.score_all(s, r, t)
        for i in range(s.size(0)):
            si,ri,oi,ti = s[i].item(),r[i].item(),o[i].item(),t[i].item()
            sc = scores[i].clone()
            for fo in filter_dict.get((si,ri,ti),set()) - {oi}:
                sc[fo] = float('-inf')
            rank = int((sc >= sc[oi]).sum().item())
            ranks.append(rank)
            if unseen_entities is not None:
                (ranks_unseen if oi in unseen_entities else ranks_seen).append(rank)

    def metrics(r):
        if not r: return {'mrr':0.0,'hits1':0.0,'hits3':0.0,'hits10':0.0}
        r = np.array(r, dtype=np.float32)
        return {'mrr':float(np.mean(1/r)),'hits1':float(np.mean(r<=1)),
                'hits3':float(np.mean(r<=3)),'hits10':float(np.mean(r<=10))}

    res = metrics(ranks)
    if unseen_entities is not None:
        res['seen']   = metrics(ranks_seen)
        res['unseen'] = metrics(ranks_unseen)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str,   required=True)
    p.add_argument('--data_dir',   type=str,   default='/storage/bhull113/rhgnn/data')
    p.add_argument('--dim',        type=int,   default=200)
    p.add_argument('--epochs',     type=int,   default=300)
    p.add_argument('--batch_size', type=int,   default=1024)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--patience',   type=int,   default=10)
    p.add_argument('--eval_every', type=int,   default=10)
    p.add_argument('--gpu',        type=int,   default=0)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    dpath  = os.path.join(args.data_dir, args.dataset)

    train = load_quadruples(os.path.join(dpath, 'train.txt'))
    valid = load_quadruples(os.path.join(dpath, 'valid.txt'))
    test  = load_quadruples(os.path.join(dpath, 'test.txt'))

    if len(valid) == 0:
        n = len(train); cut = int(n*0.9)
        idx = np.random.permutation(n)
        valid = train[idx[cut:]]; train = train[idx[:cut]]
        print(f"  No valid.txt — split: train={len(train)}, valid={len(valid)}")

    all_q  = np.concatenate([train,valid,test])
    N      = int(all_q[:,[0,2]].max()) + 1
    R      = int(all_q[:,1].max()) + 1
    t_min  = int(all_q[:,3].min())
    T      = int(all_q[:,3].max()) - t_min + 1

    # Normalize timestamps
    train[:,3] -= t_min
    valid[:,3] -= t_min
    test[:,3]  -= t_min

    train_ents = set(train[:,0].tolist() + train[:,2].tolist())
    test_ents  = set(test[:,0].tolist()  + test[:,2].tolist())
    unseen     = test_ents - train_ents

    print(f"\n{'='*50}")
    print(f"  TTransE | {args.dataset}")
    print(f"  N:{N} R:{R} T:{T} Unseen:{len(unseen)}")
    print(f"{'='*50}")

    filter_dict = build_filter(np.concatenate([train,valid,test]))
    model = TransTKG(N, R, T, args.dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)

    loader   = DataLoader(QuadDataset(train), batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    best_mrr, patience, best_state = 0.0, 0, None

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            s,r,o,t = [x.to(device) for x in batch]
            B = s.size(0)
            s_e = model.ent_emb(s)
            r_e = model.rel_emb(r)
            t_e = model.time_emb(t)
            o_e = model.ent_emb(o)
            pos = (s_e + r_e + t_e - o_e).norm(p=2, dim=-1)
            neg_o = torch.randint(0, N, (B,), device=device)
            neg   = (s_e + r_e + t_e - model.ent_emb(neg_o)).norm(p=2, dim=-1)
            loss  = torch.clamp(1.0 + pos - neg, min=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        if epoch % args.eval_every == 0:
            vr = evaluate(model, valid, filter_dict, device)
            print(f"Epoch {epoch:4d} | Loss:{total_loss/len(loader):.4f} | Val MRR:{vr['mrr']:.4f}")
            if vr['mrr'] > best_mrr:
                best_mrr = vr['mrr']; patience = 0
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(best_state)
    tr = evaluate(model, test, filter_dict, device, unseen_entities=unseen)

    print(f"\n{'='*50}")
    print(f"  TTransE Results [{args.dataset}]")
    print(f"{'='*50}")
    print(f"  MRR:{tr['mrr']:.4f}  H@1:{tr['hits1']:.4f}  H@10:{tr['hits10']:.4f}")
    if 'seen' in tr:
        print(f"  Seen:   MRR:{tr['seen']['mrr']:.4f}  H@10:{tr['seen']['hits10']:.4f}")
        print(f"  Unseen: MRR:{tr['unseen']['mrr']:.4f}  H@10:{tr['unseen']['hits10']:.4f}")
    print(f"{'='*50}")

    os.makedirs('results', exist_ok=True)
    with open(f'results/ttransE_{args.dataset}.json','w') as f:
        json.dump({'dataset':args.dataset,'test':tr},f,indent=2)
    print(f"Saved: results/ttransE_{args.dataset}.json")


if __name__ == '__main__':
    main()
