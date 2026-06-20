# KARMA-X: Anchor-Attention Encoding for Fully Inductive Temporal Knowledge Graph Reasoning

Code for the KARMA family of inductive temporal KGR models.

**Paper:**
> Bhullar, A. & Kobti, Z. (2026). *Anchor-Attention Encoding for Fully Inductive Temporal Knowledge Graph Reasoning*. ASONAM 2026 (under review).

Part of the dissertation: *Structural Selectivity in Knowledge Graph Reasoning: From Sparse Graph Neural Networks to Temporal Hyperbolic Extrapolation*. University of Windsor, 2026. Supervised by Prof. Ziad Kobti.

## Models

| Model | Key Idea | Best U-MRR |
|---|---|---|
| KARMA-base | Anchor-attention inductive encoder | 0.111 (ICEWS18) |
| KARMA-T | Transductive-inductive hybrid gate | **0.784** (WIKI-clean) |
| KARMA-A | Adaptive cluster anchors | **0.120** (ICEWS18) |
| KARMA-C | Contrastive anchor pretraining | - |
| KARMA-F | Frequency-weighted anchors | 0.772 (WIKI-clean) |
| KARMA-R | Relational path anchors | 0.769 (WIKI-clean) |
| KARMA-RL | Soft temporal rule anchors | - |
| KARMA-plus | Full stack combination | - |

## Novel Metrics

- **U-MRR**: MRR on unseen entities only
- **TGG**: Transductive-to-Generative Gap (seen MRR minus U-MRR)

## Installation

```bash
pip install torch geoopt numpy scipy
```

## Usage

```bash
cd models/KARMA-T
python train.py --dataset WIKI-clean --gpu 0 --epochs 500 --patience 20

cd models/KARMA-A
python train.py --dataset ICEWS18-renet --gpu 0 --epochs 500 --patience 20
```

## Citation

```bibtex
@inproceedings{bhullar2026karmax,
  author    = {Bhullar, Amangel and Kobti, Ziad},
  title     = {Anchor-Attention Encoding for Fully Inductive
               Temporal Knowledge Graph Reasoning},
  booktitle = {Proceedings of ASONAM 2026},
  year      = {2026},
  note      = {Under review}
}
```

## Contact

Amangel Bhullar — bhull113@uwindsor.ca
Supervisor: Prof. Ziad Kobti — kobti@uwindsor.ca
School of Computer Science, University of Windsor
