# Decoupled Epistemic Guidance (DEG)

A neuro-symbolic, architecturally decoupled safety auditor for domain-specific LLM safety.

## Overview

DEG enables practitioners to enforce institutional, regulatory, or operational constraints by authoring domain-specific safety axioms as natural-language violation patterns, without model retraining. The system computes a log-likelihood-ratio free energy functional using bidirectional NLI (DeBERTa-v3-large) aggregated via categorical LLR, combined with a recognition channel that evaluates general attack-pattern meta-hypotheses.

## Repository Structure

```
├── data/
│   ├── violation_patterns_rewritten.json   # 29 domain-specific violation patterns
│   ├── queries_all.json                     # 1,623-query benchmark (449 safe, 1,174 attack)
│   ├── meta_hypotheses.json                 # 3 recognition channel meta-hypotheses
│   └── reference_sentences.json             # 60 universal reference sentences for zero-shot calibration
├── results/
│   ├── nli_probs_cache.npz                  # Pre-computed NLI probabilities [1623, 19, 3]
│   └── paper_v2/                            # Experimental results as JSON
├── scripts/
│   ├── paper_v2_production_run.py           # Main benchmark experiment
│   ├── verify_table10.py                     # Per-domain verification
│   └── run_supplementary_experiments_v2.py   # Held-out axiom + aggregation + top-K
├── requirements.txt
└── LICENSE
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

Loads cached NLI probabilities and computes all configurations (CatLLR, raw E-C, RP-only, VP+RP) with 5-fold cross-validation.

### 2. Verify Per-Domain Results

```bash
python scripts/verify_table10.py
```

Recomputes VP CatLLR + RP scores per domain with optimal thresholds. Requires GPU (~3 minutes).

### 3. Run Supplementary Experiments

```bash
python scripts/run_supplementary_experiments_v2.py
```

Runs three experiments: held-out axiom generalization, aggregation method comparison, and top-K retrieval effect.

## Data Description

### Violation Patterns (`data/violation_patterns_rewritten.json`)

29 domain-specific violation patterns across 3 domains:
- **Nuc-Fusion** (12 VPs): Nuclear safety protocols, radiation protection, access control
- **Fin-Audit** (11 VPs): Financial compliance, authentication, data protection
- **Med-Intervene** (6 VPs): Medical ethics, pharmaceutical safety, patient privacy

### Benchmark (`data/queries_all.json`)

1,623 queries (449 safe, 1,174 adversarial) constructed through a hybrid pipeline of manual templates, LLM-assisted expansion, and human review.

### Meta-Hypotheses (`data/meta_hypotheses.json`)

3 recognition channel hypotheses selected from 12 candidates based on discriminative power.

### Reference Sentences (`data/reference_sentences.json`)

60 universal sentences (30 attack-like, 30 safe-like) for zero-shot calibration without labeled domain data.

## Key Results

| Configuration | AUC | F1 |
|--------------|-----|-----|
| NLI Direct (generic rules) | 0.619 | - |
| + Domain VPs (mean P_entail) | 0.912 | 0.904 |
| + CatLLR aggregation | 0.931 | 0.927 |
| + Recognition channel (VP+RP) | **0.945** | **0.939** |

## License

MIT License
