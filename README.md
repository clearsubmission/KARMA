# KARMA: Knowledge Anchors for Robust Memory-Augmented Inductive Temporal Knowledge Graph Reasoning

Code for the KARMA family of fully inductive temporal KGR models.

**Paper:**
> Bhullar, A. & Kobti, Z. (2026). *KARMA: Knowledge Anchors for Robust Memory-Augmented Inductive Temporal Knowledge Graph Reasoning*. IEEE ICDM 2026 (under review).

## Models

| Model | Key Contribution |
|---|---|
| KARMA-base | Anchor-attention fully inductive encoder |
| KARMA-T | Transductive-inductive hybrid gate |
| KARMA-A | Adaptive cluster anchors |
| KARMA-C | Contrastive anchor pretraining |
| KARMA-F | Frequency-weighted anchors |

## Novel Metrics

- **U-MRR**: Unseen-entity Mean Reciprocal Rank
- **TGG**: Transductive-to-Generative Gap

## Key Results (U-MRR)

| Model | YAGO15k | WIKI-clean | ICEWS14 | ICEWS18 | WIKIDATA12k | YAGO11k | ind50 | ind100 |
|---|---|---|---|---|---|---|---|---|
| KARMA-base | 0.027 | 0.018 | 0.002 | 0.111 | 0.004 | 0.003 | - | - |
| KARMA-T | 0.127 | **0.784** | 0.028 | 0.094 | 0.001 | 0.000 | - | - |
| KARMA-A | 0.000 | 0.042 | 0.018 | **0.120** | 0.002 | 0.000 | - | - |
| KARMA-C | - | - | - | - | - | - | - | - |
| KARMA-F | 0.041 | 0.772 | 0.007 | 0.101 | 0.002 | 0.000 | - | - |

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

- YAGO15k
- WIKI-clean
- WIKIDATA12k
- ICEWS14-renet
- ICEWS18-renet
- YAGO11k

## Citation

```bibtex
@inproceedings{bhullar2026karma,
  author    = {Bhullar, Amangel and Kobti, Ziad},
  title     = {KARMA: Knowledge Anchors for Robust Memory-Augmented
               Inductive Temporal Knowledge Graph Reasoning},
  booktitle = {Proceedings of IEEE ICDM 2026},
  year      = {2026},
  note      = {Under review}
}
```
