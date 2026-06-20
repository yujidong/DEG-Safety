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
│   ├── queries_all.json                     # 1,623-query benchmark (449 safe, 1,174 attack)
│   ├── meta_hypotheses.json                 # 3 recognition channel meta-hypotheses
│   └── reference_sentences.json             # 60 universal reference sentences
├── results/                                # Pre-computed results
│   ├── nli_probs_cache.npz                  # Cached NLI probabilities
│   └── paper_v2/                            # Result JSON files for each experiment
├── examples/
│   └── basic_usage.py                       # Quick start example
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

### Violation Patterns (`data/violation_patterns_rewritten.json`)

29 domain-specific violation patterns across 3 domains:
- **Nuc-Fusion** (12 VPs): Nuclear safety protocols, radiation protection, access control
- **Fin-Audit** (11 VPs): Financial compliance, authentication, data protection
- **Med-Intervene** (6 VPs): Medical ethics, pharmaceutical safety, patient privacy

### Benchmark (`data/queries_all.json`)

1,623 queries (449 safe, 1,174 adversarial) with 8 attack vector types: Context Flooding, Few-Shot Poisoning, Logic Nesting, Social Engineering, Prompt Injection, JSON Jailbreak, Cognitive Overload, and Rhetorical Camouflage.

## License

MIT License
