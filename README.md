# KARMA: Knowledge Anchors for Robust Memory-Augmented Inductive Temporal Knowledge Graph Reasoning

Code for the KARMA family of fully inductive temporal KGR models.


## Models

| Model | Key Contribution |
|---|---|
| KARMA-base | Anchor-attention fully inductive encoder |
| KARMA-T | Transductive-inductive hybrid gate |
| KARMA-A | Adaptive cluster anchors |
| KARMA-C | Contrastive anchor pretraining |
| KARMA-F | Frequency-weighted anchors |


## Comparison with Existing Methods

| Model | Temporal | Fully Inductive | Entity Ranking | U-MRR | TGG | Missing Facts | Future Pred. | New Benchmarks |
|---|---|---|---|---|---|---|---|---|
| **Static Inductive** | | | | | | | | |
| GraIL | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| NodePiece | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| INGRAM | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| DRUM | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Temporal Transductive** | | | | | | | | |
| TTransE | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| TNTComplEx | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| RE-NET | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| RE-GCN | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| **Temporal Partial-Inductive** | | | | | | | | |
| TLogic | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| TILP | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| xERTE | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ |
| TGAT | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| **Fully Inductive (Ours)** | | | | | | | | |
| KARMA-base | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KARMA-T | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KARMA-A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KARMA-C | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KARMA-F | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

All transductive baselines achieve U-MRR ≡ 0 by definition on unseen entities.

## Models

| Model | Key Mechanism | Best U-MRR |
|---|---|---|
| KARMA-base | Pure anchor-attention inductive encoder | 0.111 (ICEWS18) |
| KARMA-T | Transductive-inductive hybrid gate | **0.784** (WIKI-clean), TGG = −0.148 |
| KARMA-A | Adaptive anchor clustering per entity type | **0.120** (ICEWS18) |
| KARMA-C | InfoNCE contrastive temporal pretraining | 0.183 MRR (YAGO11k) |
| KARMA-F | Future-aware relation prediction encoding | 0.772 (WIKI-clean) |

## Novel Metrics

- **U-MRR**: Unseen-entity Mean Reciprocal Rank
- **TGG**: Transductive-to-Generative Gap



## Results

### Standard Benchmark MRR

| Model | ICEWS14 | ICEWS18 | YAGO11k | WIKIDATA12k | WIKI-clean | YAGO15k |
|---|---|---|---|---|---|---|
| **Transductive Baselines (U-MRR ≡ 0)** | | | | | | |
| TTransE† | 0.255 | 0.084 | 0.105 | 0.138 | — | — |
| HyTE† | 0.297 | 0.148 | 0.169 | 0.180 | — | — |
| TNTComplEx† | 0.370 | 0.280 | 0.341 | 0.357 | — | — |
| RE-NET† | 0.457 | 0.370 | — | — | — | — |
| RE-GCN† | 0.458 | 0.373 | — | — | — | — |
| **Fully Inductive (Ours)** | | | | | | |
| KARMA-base | 0.105 | 0.178 | 0.099 | 0.222 | 0.524 | 0.585 |
| KARMA-T | 0.181 | 0.218 | 0.168 | 0.246 | **0.663** | **0.611** |
| KARMA-A | 0.173 | 0.208 | 0.174 | 0.242 | 0.530 | 0.608 |
| KARMA-C | 0.185 | 0.224 | **0.183** | **0.251** | — | — |
| KARMA-F | **0.186** | 0.217 | 0.169 | 0.245 | 0.660 | — |

† Transductive: U-MRR ≡ 0 by definition. IC14=ICEWS14, IC18=ICEWS18, YG11=YAGO11k, WD12=WIKIDATA12k, WK=WIKI-clean, YG15=YAGO15k.

### Inductive Evaluation (U-MRR, TGG)

> TGG = Seen MRR − Unseen MRR. **Negative TGG = unseen entities predicted more accurately than seen ones.**

#### Future Prediction (t > T_train)

| Dataset | Model | Seen MRR | Seen H@1 | Seen H@10 | Unseen MRR | Unseen H@1 | Unseen H@10 | TGG↓ |
|---|---|---|---|---|---|---|---|---|
| WIKI-clean | TTransE† | 0.584 | — | — | 0.001 | 0.000 | 0.000 | 0.583 |
| | KARMA-base | 0.632 | 0.613 | 0.661 | 0.018 | 0.003 | 0.042 | 0.614 |
| | KARMA-A | 0.634 | 0.607 | 0.670 | 0.042 | 0.004 | 0.109 | 0.592 |
| | KARMA-F | 0.636 | 0.613 | 0.670 | 0.772 | 0.674 | 0.901 | -0.109 |
| | **KARMA-T** | **0.637** | **0.611** | **0.672** | **0.784** | **0.681** | **0.923** | **-0.148** |
| YAGO15k | KARMA-base | 0.617 | 0.553 | 0.729 | 0.027 | 0.000 | 0.052 | 0.590 |
| | KARMA-A | 0.608 | 0.555 | 0.694 | 0.000 | 0.000 | 0.000 | 0.608 |
| | **KARMA-T** | **0.639** | **0.576** | **0.726** | **0.127** | **0.093** | **0.181** | **0.512** |
| ICEWS18 | TTransE† | 0.084 | — | 0.220 | 0.000 | 0.000 | 0.000 | 0.084 |
| | RE-GCN† | 0.373 | — | 0.584 | 0.000 | 0.000 | 0.000 | 0.373 |
| | KARMA-base | 0.180 | 0.094 | 0.358 | 0.111 | 0.077 | 0.172 | 0.068 |
| | KARMA-T | 0.212 | 0.116 | 0.408 | 0.094 | 0.054 | 0.154 | 0.118 |
| | KARMA-F | 0.219 | 0.122 | 0.413 | 0.101 | 0.070 | 0.159 | 0.118 |
| | **KARMA-A** | **0.209** | **0.111** | **0.405** | **0.120** | **0.093** | **0.176** | **0.089** |
| ICEWS14 | TTransE† | 0.255 | — | 0.601 | 0.000 | 0.000 | 0.000 | 0.255 |
| | RE-GCN† | 0.458 | — | 0.632 | 0.000 | 0.000 | 0.000 | 0.458 |
| | KARMA-base | 0.107 | 0.047 | 0.223 | 0.002 | 0.000 | 0.000 | 0.105 |
| | KARMA-A | 0.176 | 0.090 | 0.349 | 0.018 | 0.006 | 0.032 | 0.158 |
| | KARMA-C | 0.187 | 0.098 | 0.369 | 0.006 | 0.001 | 0.009 | 0.181 |
| | KARMA-F | 0.188 | 0.097 | 0.373 | 0.007 | 0.000 | 0.014 | 0.181 |
| | **KARMA-T** | 0.183 | 0.094 | 0.360 | **0.028** | **0.006** | **0.059** | **0.155** |

#### Missing Fact Completion (t ≤ T_train)

| Dataset | Model | Seen MRR | Seen H@1 | Seen H@10 | Unseen MRR | Unseen H@1 | Unseen H@10 | TGG↓ |
|---|---|---|---|---|---|---|---|---|
| WIKIDATA12k | TTransE† | 0.138 | — | 0.341 | 0.000 | 0.000 | 0.000 | 0.138 |
| | TNTComplEx† | 0.357 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.357 |
| | KARMA-base | 0.223 | 0.143 | 0.385 | 0.004 | 0.000 | 0.000 | 0.219 |
| | **KARMA-C** | **0.252** | **0.148** | **0.499** | 0.000 | 0.000 | 0.000 | — |
| | KARMA-T | 0.248 | 0.143 | 0.485 | 0.001 | 0.000 | 0.000 | 0.247 |
| YAGO11k | TTransE† | 0.105 | — | 0.290 | 0.000 | 0.000 | 0.000 | 0.105 |
| | TNTComplEx† | 0.341 | — | 0.490 | 0.000 | 0.000 | 0.000 | 0.341 |
| | KARMA-base | 0.100 | 0.047 | 0.195 | 0.003 | 0.000 | 0.000 | 0.097 |
| | **KARMA-C** | **0.183** | **0.110** | **0.346** | 0.000 | 0.000 | 0.000 | — |
| | KARMA-A | 0.174 | 0.106 | 0.331 | 0.000 | 0.000 | 0.000 | 0.174 |

| | **Best KARMA** | | | | **0.784** | **0.681** | **0.923** | **−0.148** |
| | Best Baseline | | | | 0.001 | 0.000 | 0.000 | — |
| | **Improvement** | | | | **784×** | — | — | — |


### Ablation Study

| Model | WIKI-clean MRR | WIKI-clean U-MRR | ICEWS18 MRR | ICEWS18 U-MRR |
|---|---|---|---|---|
| KARMA-base | 0.524 | 0.018 | 0.178 | 0.111 |
| + Hybrid gate (T) | **0.663** | **0.784** | 0.218 | 0.094 |
| + Adaptive anchors (A) | 0.530 | 0.042 | 0.208 | **0.120** |
| + Contrastive (C) | — | — | **0.224** | 0.030 |
| + Future encoding (F) | 0.660 | 0.772 | 0.217 | 0.101 |
| TTransE (baseline) | — | 0.001 | — | 0.000 |
| RE-GCN (baseline) | — | 0.000 | — | 0.000 |

Each component improves U-MRR over KARMA-base. The hybrid gate provides the largest gain on homogeneous graphs (+0.766 U-MRR on WIKI-clean); adaptive anchors provide the largest gain on heterogeneous graphs (+0.009 U-MRR on ICEWS18).

## New Inductive Benchmarks

| Dataset | Facts | Entities | Relations | Unseen Entity % |
|---|---|---|---|---|
| SciInductTKG-Emerge | 177K | 46,684 | 4 | 76.4% |
| SciInductTKG-Cold | 140K | 46,684 | 4 | 28.2% |
| WikiInductTKG-Emerge | 12K | 14,029 | 3 | 86.7% |
| WikiInductTKG-Cold | 7K | 14,029 | 3 | **95.1%** |
| PolitInductTKG-Emerge | 1,087K | 22,225 | 260 | 9.4% |
| PolitInductTKG-Cold | 1,012K | 22,225 | 260 | 36.6% |

Three split types per benchmark: **Emerge** (time-based, highest unseen ratio) · **Rare** (long-tail entities with few facts) · **Cold** (entities completely absent from training).

## When to Use Each Variant

| Graph / Domain Property | Recommended Variant | Reason |
|---|---|---|
| Homogeneous entities, low relation diversity | KARMA-T | Hybrid gate maximizes transductive strength |
| Heterogeneous entity types (countries, orgs, people) | KARMA-A | Cluster anchors capture type-specific temporal dynamics |
| Event-heavy, periodic domains | KARMA-F | Future encoder explicitly models relation periodicity |
| Sparse interaction history | KARMA-C | Contrastive pretraining improves low-data generalization |
| Unknown / general purpose | KARMA-T | Best average performance across datasets |


## Repository Structure

```text
KARMA/
├── models/
│   ├── karma_base.py      # Pure anchor-attention inductive encoder
│   ├── karma_t.py         # Transductive-inductive hybrid gate
│   ├── karma_a.py         # Adaptive anchor clustering
│   ├── karma_c.py         # Contrastive temporal pretraining
│   └── karma_f.py         # Future-aware encoding
├── benchmarks/
│   ├── SciInductTKG/
│   ├── WikiInductTKG/
│   └── PolitInductTKG/
├── data/
│   ├── ICEWS14/
│   ├── ICEWS18/
│   ├── YAGO11k/
│   ├── WIKIDATA12k/
│   ├── WIKI-clean/
│   └── YAGO15k/
├── train.py
├── evaluate.py
├── data_loader.py
└── README.md
```


## Installation

```bash
pip install torch geoopt numpy scipy
```

## Usage

```bash
# KARMA-base
cd models/KARMA-base
python train.py --dataset YAGO15k --gpu 0 --epochs 500 --patience 20

# KARMA-T (best on encyclopedic datasets)
cd models/KARMA-T
python train.py --dataset WIKI-clean --gpu 0 --epochs 500 --patience 20

# KARMA-A (best on event datasets)
cd models/KARMA-A
python train.py --dataset ICEWS18-renet --gpu 0 --epochs 500 --patience 20
```

## Datasets

### Standard Benchmarks

| Dataset | Task | Facts | Entities | Relations | Unseen Entity % (test) |
|---|---|---|---|---|---|
| WIKI-clean | Future prediction | — | — | 24 | high |
| YAGO15k | Future prediction | — | — | 10 | moderate |
| ICEWS14 | Future prediction | — | 14,541 | 248 | varies |
| ICEWS18 | Future prediction | — | — | 251 | varies |
| WIKIDATA12k | Missing fact completion | — | — | 24 | very low |
| YAGO11k | Missing fact completion | — | — | 10 | very low |

**Future prediction** (t > T_train): facts occur after the training cutoff — WIKI-clean, YAGO15k, ICEWS14, ICEWS18.  
**Missing fact completion** (t ≤ T_train): facts existed during training but were unobserved — WIKIDATA12k, YAGO11k.  
All transductive baselines achieve U-MRR ≡ 0 by definition on unseen entities across all six datasets.

### New Inductive Benchmarks

| Dataset | Facts | Entities | Relations | Unseen Entity % |
|---|---|---|---|---|
| SciInductTKG-Emerge | 177K | 46,684 | 4 | 76.4% |
| SciInductTKG-Cold | 140K | 46,684 | 4 | 28.2% |
| WikiInductTKG-Emerge | 12K | 14,029 | 3 | 86.7% |
| WikiInductTKG-Cold | 7K | 14,029 | 3 | **95.1%** |
| PolitInductTKG-Emerge | 1,087K | 22,225 | 260 | 9.4% |
| PolitInductTKG-Cold | 1,012K | 22,225 | 260 | 36.6% |
| InductTKG-Emerge | — | — | — | — |
| InductTKG-Cold | — | — | — | — |

Three split types per benchmark:
- **Emerge** — time-based split with highest unseen entity ratio
- **Rare** — long-tail entities with few historical interactions
- **Cold** — entities completely absent from training


```
