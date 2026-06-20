"""
DEG (Decoupled Epistemic Guidance) - A Cognitive Architecture for Axiomatic Safety in LLMs

Based on the paper: "From Likelihood to Free Energy: A Cognitive Architecture for Axiomatic Safety in LLMs"
"""

__version__ = "0.1.0"

# Main framework
from .architecture import DEGFramework, create_full_deg, create_deg_with_encoder, DEGDecision

# Phase I: Intent extraction
from .intent_extractor import IntentExtractor, NLIIntentExtractor, LLMIntentExtractor
from .intent_extractor_encoder import IntentExtractorEncoder, HybridIntentExtractor, ACTION_ONTOLOGY, SUSPICION_PATTERNS

# Phase I+: Multi-intent extraction (NEW)
from .multi_intent_extractor import (
    BaseMultiIntentExtractor,
    MultiIntentResult,
    UncertaintyGuidedExtractor,
    create_multi_intent_extractor
)

# DEG Factory with multi-intent support (NEW)
from .deg_factory import MultiIntentDEGFramework, create_deg_multi_intent

# Phase II: Axiomatic graph
from .axiomatic_graph import AxiomaticGraph, LocalAxiomaticGraph, Axiom

# Phase III: Energy calculation
from .energy_calculator import (
    EnergyCalculator,
    HeuristicEnergyCalculator,
    NLIEnergyCalculator,
    EnergyResult,
    BAYESIAN_ALPHA_DEFAULT,
    BAYESIAN_BETA_DEFAULT,
    META_HYPOTHESES,
)

# Phase IV: Decision engine
from .decision_engine import (
    DecisionEngine,
    AdaptiveDecisionEngine,
    SimpleDecisionEngine,
    Action,
    Decision,
    PolicyDecision
)

__all__ = [
    # Main framework
    "DEGFramework",
    "create_full_deg",
    "create_deg_with_encoder",
    "DEGDecision",
    # Decision types
    "EnergyResult",
    "PolicyDecision",
    # Phase I: Intent extractors
    "IntentExtractor",
    "NLIIntentExtractor",
    "LLMIntentExtractor",
    "IntentExtractorEncoder",
    "HybridIntentExtractor",
    "ACTION_ONTOLOGY",
    "SUSPICION_PATTERNS",
    # Phase I+: Multi-intent extraction
    "BaseMultiIntentExtractor",
    "MultiIntentResult",
    "UncertaintyGuidedExtractor",
    "create_multi_intent_extractor",
    # Multi-intent DEG
    "MultiIntentDEGFramework",
    "create_deg_multi_intent",
    # Phase II: Axiomatic graphs
    "AxiomaticGraph",
    "LocalAxiomaticGraph",
    "Axiom",
    # Phase III: Energy calculators
    "EnergyCalculator",
    "HeuristicEnergyCalculator",
    "NLIEnergyCalculator",
    "BAYESIAN_ALPHA_DEFAULT",
    "BAYESIAN_BETA_DEFAULT",
    "META_HYPOTHESES",
    # Phase IV: Decision engine
    "DecisionEngine",
    "AdaptiveDecisionEngine",
    "SimpleDecisionEngine",
    "Action",
    "Decision",
]
