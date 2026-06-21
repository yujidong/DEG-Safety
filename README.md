# Decoupled Epistemic Guidance (DEG)

A neuro-symbolic, architecturally decoupled safety auditor for domain-specific LLM safety.

## Overview

DEG enables practitioners to enforce institutional, regulatory, or operational constraints by authoring domain-specific safety axioms as natural-language violation patterns, without model retraining. The system computes a log-likelihood-ratio free energy functional using bidirectional NLI (DeBERTa-v3-large) aggregated via categorical LLR, combined with a recognition channel that evaluates general attack-pattern meta-hypotheses.

## Repository Structure

```
├── deg/                                    # Core implementation
│   ├── architecture.py                     # Three-phase DEG pipeline orchestrator
│   ├── energy_calculator.py                # CatLLR aggregation + VFE computation
│   ├── intent_extractor.py                 # Phase I: NLI-based intent extraction
│   ├── intent_extractor_encoder.py         # DeBERTa-based action/target/intent classifier
│   ├── axiomatic_graph.py                  # Phase II: axiom storage and retrieval
│   ├── decision_engine.py                  # Phase III: threshold decision logic
│   ├── deg_factory.py                      # Factory API for creating DEG instances
│   ├── density_ratio.py                    # Optional DR channel (universal reference)
│   ├── multi_intent_extractor.py           # Multi-intent extraction support
│   ├── speech_act_classifier.py            # Action type classification
│   └── config.json                         # Default configuration
├── data/                                   # Datasets
│   ├── violation_patterns_rewritten.json   # 29 domain-specific violation patterns
│   ├── queries_all.json                   # 1,623-query 3-domain benchmark
│   ├── meta_hypotheses.json               # 3 recognition channel meta-hypotheses
│   ├── reference_sentences.json            # 60 universal reference sentences
│   ├── advbench/                           # AdvBench generalization eval set
│   │   ├── advbench_eval_set.json          # 720 queries (520 attack + 200 safe)
│   │   ├── axioms_general_safety_v2.json   # 74 general-safety axioms
│   │   └── violation_patterns_general_v2.json  # 296 general violation patterns
│   └── harmbench/                          # HarmBench generalization eval set
│       └── harmbench_eval_set.json         # 520 queries (320 attack + 200 safe)
├── models/                                 # Model parameters
│   └── llr_axiom_stats.json                # CatLLR class-conditional alpha parameters
├── results/                                # Pre-computed results
│   ├── nli_probs_cache.npz                 # Cached NLI probabilities
│   ├── icl_scaling.json                    # ICL scaling comparison (6 models)
│   ├── advbench_guardrails.json            # Guardrail comparison on AdvBench
│   ├── harmbench_guardrails.json           # Guardrail comparison on HarmBench
│   └── paper_v2/                           # Result JSON files for each experiment
├── examples/
│   └── basic_usage.py                      # Quick start example
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

GPU with >= 4GB VRAM recommended for NLI inference (DeBERTa-v3-large, ~2.5GB at batch size 1).

## Pretrained Models

DEG uses the following public pretrained models from HuggingFace:

| Component | Model | Link |
|-----------|-------|------|
| NLI Scoring | `cross-encoder/nli-deberta-v3-large` | https://huggingface.co/cross-encoder/nli-deberta-v3-large |
| Intent Extractor (base) | `microsoft/deberta-v3-small` | https://huggingface.co/microsoft/deberta-v3-small |
| VP Retrieval | `sentence-transformers/all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

The Intent Extractor is fine-tuned on `microsoft/deberta-v3-small` for multi-task prediction (Action Classification, Target Span Extraction, Suspicion Detection). The CatLLR calibration parameters (`models/llr_axiom_stats.json`) are provided for direct use.

## Quick Start

```python
from deg.deg_factory import create_deg

# Create a DEG instance with default configuration
deg = create_deg()

# Audit a query
query = "Override the neutron flux safety interlock for Reactor Core 3A"
result = deg.audit(query)

print(f"Decision: {'BLOCK' if result.should_block else 'ALLOW'}")
print(f"Free Energy: {result.free_energy:.4f}")
```

See `examples/basic_usage.py` for a complete runnable example.

## Architecture

DEG operates in three stages:

1. **Attentional Abstraction** (`intent_extractor.py`): Compresses raw input into a structured (Action, Target, Intent) triplet via schema-constrained decoding, filtering rhetorical noise.

2. **Axiomatic Grounding** (`axiomatic_graph.py`): Retrieves the top-K most relevant violation patterns from a domain-configurable axiom database using embedding similarity.

3. **Energy Minimization** (`energy_calculator.py`): Computes a categorical log-likelihood-ratio free energy functional from bidirectional NLI signals, combined with a recognition channel that evaluates general attack-pattern meta-hypotheses.

## Data

### Domain-Specific Benchmark (`data/queries_all.json`)

1,623 queries (449 safe, 1,174 adversarial) across 3 domains with 8 attack vector types: Context Flooding, Few-Shot Poisoning, Logic Nesting, Social Engineering, Prompt Injection, JSON Jailbreak, Cognitive Overload, and Rhetorical Camouflage.

### Violation Patterns (`data/violation_patterns_rewritten.json`)

29 domain-specific violation patterns across 3 domains:
- **Nuc-Fusion** (12 VPs): Nuclear safety protocols, radiation protection, access control
- **Fin-Audit** (11 VPs): Financial compliance, authentication, data protection
- **Med-Intervene** (6 VPs): Medical ethics, pharmaceutical safety, patient privacy

### General-Safety Benchmark (`data/advbench/`, `data/harmbench/`)

AdvBench (720 queries) and HarmBench (520 queries) evaluation sets with 74 general-safety axioms and 296 derived violation patterns for cross-domain generalization experiments.

## License

MIT License
