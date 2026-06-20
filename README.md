# Decoupled Epistmic Guidance (DEG)

Code and data for the paper: **"From Likelihood to Free Energy: A Cognitive Architecture for Axiomatic Safety in Large Language Models"**

## Overview

DEG is a neuro-symbolic, architecturally decoupled safety auditor for domain-specific LLM safety. It enables practitioners to enforce institutional, regulatory, or operational constraints by authoring domain-specific safety axioms as natural-language violation patterns, without model retraining.

The system computes a log-likelihood-ratio free energy functional using bidirectional NLI (DeBERTa-v3-large) aggregated via categorical LLR, combined with a recognition channel that evaluates general attack-pattern meta-hypotheses.

## Repository Structure

```
deg-public-release/
├── data/
│   ├── violation_patterns_rewritten.json   # 29 domain-specific violation patterns
│   ├── queries_all.json                     # 1,623-query benchmark (449 safe, 1,174 attack)
│   ├── meta_hypotheses.json                 # 3 recognition channel meta-hypotheses
│   └── reference_sentences.json             # 60 universal reference sentences for zero-shot calibration
├── results/
│   ├── nli_probs_cache.npz                  # Pre-computed NLI probabilities [1623, 19, 3]
│   └── paper_v2/                            # All experimental results as JSON
│       ├── table01_main_benchmark.json      # Main results (Table 1/2)
│       ├── table10_corrected.json            # Per-domain results (Table 10)
│       ├── supplementary_experiments.json    # Held-out axiom, aggregation comparison, top-K
│       └── ...                               # 20+ result files
├── scripts/
│   ├── paper_v2_production_run.py           # Main benchmark experiment
│   ├── verify_table10.py                     # Per-domain verification
│   └── run_supplementary_experiments_v2.py   # Held-out axiom + aggregation + top-K
├── README.md
└── requirements.txt
```

## Requirements

```
torch>=2.0
transformers>=4.36
sentence-transformers>=2.2
scikit-learn>=1.3
numpy>=1.24
scipy>=1.10
```

GPU with >= 8GB VRAM required for NLI inference (DeBERTa-v3-large).

## Quick Start

### 1. Reproduce Main Results

```bash
python scripts/paper_v2_production_run.py
```

This loads the cached NLI probabilities and computes all configurations (CatLLR, raw E-C, RP-only, VP+RP) with 5-fold cross-validation. Output is saved to `results/paper_v2/`.

### 2. Verify Per-Domain Results

```bash
python scripts/verify_table10.py
```

Recomputes VP CatLLR + RP scores per domain with optimal thresholds. Requires GPU for RP NLI inference (~3 minutes).

### 3. Run Supplementary Experiments

```bash
python scripts/run_supplementary_experiments_v2.py
```

Runs three experiments:
- **Held-out axiom**: Configures DEG with 14 of 29 VPs, measures recall on attacks targeting held-out axioms
- **Aggregation comparison**: Compares mean P(entail), raw E-C, CatLLR on same domain VPs
- **Top-K retrieval**: Compares CatLLR performance under varying K (3, 5, 8, all)

## Data Description

### Violation Patterns (`violation_patterns_rewritten.json`)

29 domain-specific violation patterns across 3 domains:
- **Nuc-Fusion** (12 VPs): Nuclear safety protocols, radiation protection, access control
- **Fin-Audit** (11 VPs): Financial compliance, authentication, data protection
- **Med-Intervene** (6 VPs): Medical ethics, pharmaceutical safety, patient privacy

Each VP is a natural-language statement describing the consequence of violating a domain axiom, evaluated by the NLI model against the query.

### Benchmark (`queries_all.json`)

1,623 queries (449 safe, 1,174 adversarial) constructed through a hybrid pipeline:
- Attack templates drafted manually for 8 attack vectors
- Expanded using Qwen-2.5-72B via Ollama
- Manual review and filtering for quality and axiom-target alignment
- Safe queries from domain documentation and textbook-style questions

### Meta-Hypotheses (`meta_hypotheses.json`)

3 recognition channel hypotheses selected from 12 candidates based on per-hypothesis discriminative power on the development set.

### Reference Sentences (`reference_sentences.json`)

60 universal sentences (30 attack-like, 30 safe-like) for zero-shot CatLLR calibration without labeled domain data.

## Key Results

| Configuration | AUC | F1 |
|--------------|-----|-----|
| NLI Direct (generic rules) | 0.619 | - |
| + Domain VPs (mean P_entail) | 0.912 | 0.904 |
| + CatLLR aggregation | 0.931 | 0.927 |
| + Recognition channel (VP+RP) | **0.945** | **0.939** |

Progressive decomposition: domain axiom content contributes +29.3 pp AUC, CatLLR +1.9 pp, channel complementarity +1.4 pp.

## Citation

```bibtex
@article{dong2025deg,
  title={From Likelihood to Free Energy: A Cognitive Architecture for Axiomatic Safety in Large Language Models},
  author={Dong, Yuji},
  journal={Neurocomputing},
  year={2025}
}
```

## License

MIT License
