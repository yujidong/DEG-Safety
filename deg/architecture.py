"""
DEG Framework - Main Architecture

Orchestrates the four-phase pipeline:
1. Intent Extraction (Phase I)
2. Axiomatic Retrieval (Phase II)
3. Energy Calculation (Phase III)
4. Decision Engine (Phase IV)
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import os

from .intent_extractor import IntentExtractor, NLIIntentExtractor, LLMIntentExtractor
from .intent_extractor_encoder import IntentExtractorEncoder, HybridIntentExtractor, ACTION_ONTOLOGY, SUSPICION_PATTERNS
from .axiomatic_graph import AxiomaticGraph, LocalAxiomaticGraph, Axiom
from .energy_calculator import EnergyCalculator, NLIEnergyCalculator, EnergyResult
from .decision_engine import DecisionEngine, AdaptiveDecisionEngine, SimpleDecisionEngine, Action, Decision, PolicyDecision


@dataclass
class DEGDecision:
    """
    Decision output from the DEG framework.

    Attributes:
        safe: Whether the request is deemed safe
        energy_result: Full energy calculation details
        extracted_intent: The extracted intent
        reasoning: Human-readable explanation
    """
    safe: bool
    energy_result: EnergyResult
    extracted_intent: Dict[str, Any]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "total_energy": self.energy_result.total_energy,
            "pragmatic_value": self.energy_result.pragmatic_value,
            "epistemic_value": self.energy_result.epistemic_value,
            "intent": self.extracted_intent,
            "reasoning": self.reasoning
        }


class DEGFramework:
    """
    Decoupled Epistemic Guidance (DEG) Framework.

    A neuro-symbolic architecture that operates as an external metacognitive
    regulator, structurally orthogonal to the host generative model.
    """

    def __init__(
        self,
        intent_extractor: Optional[IntentExtractor] = None,
        axiomatic_graph: Optional[AxiomaticGraph] = None,
        energy_calculator: Optional[EnergyCalculator] = None,
        decision_engine: Optional[DecisionEngine] = None,
        threshold: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the DEG framework.

        Args:
            intent_extractor: Phase I - Intent extraction module
            axiomatic_graph: Phase II - Axiomatic graph
            energy_calculator: Phase III - Energy calculator
            decision_engine: Phase IV - Decision engine
            threshold: Homeostatic threshold for blocking (legacy, for simple engine)
            verbose: Whether to print detailed reasoning
        """
        # IMPORTANT: Use NLIIntentExtractor by default (theoretically aligned)
        # NLI-based dual extraction implements consequence-based contradiction detection
        self.intent_extractor = intent_extractor or NLIIntentExtractor(
            llm_model="Qwen/Qwen2.5-1.5B-Instruct"
        )

        # Load domain-specific axioms by default
        if axiomatic_graph is None:
            import os
            axioms_file = os.path.join(
                os.path.dirname(__file__),
                "..",
                "benchmark",
                "data",
                "phase1",
                "axioms_all_tagged.json"
            )
            self.axiomatic_graph = LocalAxiomaticGraph(
                embedding_model="all-MiniLM-L6-v2"
            )
            if os.path.exists(axioms_file):
                self.axiomatic_graph.load_from_file(axioms_file)
                if verbose:
                    print(f"[DEG] Loaded {len(self.axiomatic_graph.axioms)} axioms from {axioms_file}")
            else:
                if verbose:
                    print(f"[DEG] Warning: Using {len(self.axiomatic_graph.axioms)} default axioms")
        else:
            self.axiomatic_graph = axiomatic_graph
        self.energy_calculator = energy_calculator or NLIEnergyCalculator(
            threshold=threshold
        )
        # Use the provided threshold for decision engine
        # If no decision_engine provided, create SimpleDecisionEngine with the threshold
        if decision_engine is None:
            self.decision_engine = SimpleDecisionEngine({
                "threshold": threshold  # Use the provided threshold
            })
        else:
            self.decision_engine = decision_engine

        self.threshold = threshold
        self.verbose = verbose

    def preload(self):
        """
        Preload all lazy-loaded models to eliminate cold start latency.

        Call this method before running benchmarks to ensure all models are loaded:
        - Axiomatic graph embedding model (sentence-transformers)
        - Energy calculator NLI model (DeBERTa-v3)

        This eliminates the ~7-10 second cold start on first query.
        """
        # Preload axiomatic graph embedding model
        if hasattr(self.axiomatic_graph, 'preload'):
            self.axiomatic_graph.preload()

        # Preload energy calculator NLI model
        if hasattr(self.energy_calculator, 'preload'):
            self.energy_calculator.preload()

    def audit(self, user_input: str) -> DEGDecision:
        """
        Audit a user input through the four-phase DEG pipeline.

        Args:
            user_input: Raw user input text

        Returns:
            DEGDecision with safety assessment
        """
        # Phase I: Extract intent with action type
        # OPTIMIZATION: Call extract() ONCE instead of extract_with_type() + extract_with_energy()
        # This avoids duplicate LLM inference, reducing latency from 12s to 6s
        if self.verbose:
            print("Phase I: Intent Extraction (with action type)")
        intent = self.intent_extractor.extract(user_input)

        # Extract action_type and epistemic_entropy from single extraction result
        action_type = intent.get("type", intent.get("action_type", "READ"))

        # Compute epistemic entropy H(Q(z)) from posterior confidence
        # Corrected: use posterior entropy H(Q) = -p*log(p) - (1-p)*log(1-p),
        # not the embedding standard deviation (which was a proxy for H(P), the prior)
        import numpy as np
        confidence = intent.get("overall_confidence", 0.5)
        p = max(min(confidence, 1.0 - 1e-10), 1e-10)
        epistemic_entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

        # Phase II: Retrieve relevant axioms
        if self.verbose:
            print("Phase II: Axiomatic Retrieval")
        # Use configurable top_k (can be set via axiomatic_graph.retrieve_relevant parameter)
        # Default to 15 for optimal recall/precision balance
        # Pass speech_act for pragmatic precision modulation (filter axioms by speech act relevance)
        # Pass action_type to resolve pragmatics conflicts (COMMAND + READ -> QUESTION filtering)
        speech_act = intent.get("speech_act")
        relevant_axioms = self.axiomatic_graph.retrieve_relevant(intent, top_k=15, speech_act=speech_act, action_type=action_type)

        # Phase III: Compute free energy
        if self.verbose:
            print("Phase III: Energy Calculation")
        energy_result = self.energy_calculator.compute_free_energy(
            intent,
            relevant_axioms,
            epistemic_entropy
        )

        # Phase IV: Decision (NEW)
        if self.verbose:
            print("Phase IV: Decision Engine")

        # Find most violated axiom (if any)
        violated_axiom = None
        if energy_result.details.get("axiom_violations"):
            violated_axiom = energy_result.details["axiom_violations"][0].get("axiom_id")

        # Suspicious pattern threshold adjustment
        # THEORETICAL BASIS: When deceptive framing is detected (e.g., legitimacy_masking),
        # the prior should shift toward higher risk. This implements Bayesian belief revision:
        # P(attack|legitimacy_masking) > P(attack), so the decision threshold should be
        # lowered to increase sensitivity. This is analogous to lowering the criterion in
        # signal detection theory when the base rate of the signal increases.
        suspicious_patterns = intent.get("suspicious_patterns", [])
        consequence_text = intent.get("consequence", "").lower().strip()
        safe_consequences = ["no state change", "no change", "no modification",
                             "informational only", "read only"]
        has_deceptive_framing = (
            any(p in suspicious_patterns for p in ['legitimacy_masking', 'authority_exploitation',
                                                     'compliance_abuse', 'policy_manipulation'])
            and consequence_text in safe_consequences
        )

        # Get decision from decision engine
        # Use consequence (state change) to determine risk level, not just action_type.
        # "Generate report" is WRITE but has no state change → use READ (lenient) threshold.
        # "Disable safety" is WRITE with state change → use WRITE (strict) threshold.
        is_state_change = consequence_text not in safe_consequences and action_type == "WRITE"
        action_enum = Action.WRITE if is_state_change else Action.READ

        # When deceptive framing is detected with uninformative consequence,
        # use a lowered threshold (2.0 instead of 2.5 for READ)
        # This is safe because legitimacy_masking never triggers on genuine SAFE queries
        energy_for_decision = energy_result.total_energy
        if has_deceptive_framing and action_enum == Action.READ:
            # Lowered threshold for legitimacy-masked queries
            # Standard READ threshold is 2.5; with deceptive framing, use 2.0
            policy_decision = self.decision_engine.decide(
                energy_for_decision,
                action_enum,
                violated_axiom
            )
            # Override: if energy > 2.0, block even with READ classification
            if energy_for_decision > 2.0 and policy_decision.decision == Decision.COLLAB:
                from .decision_engine import PolicyDecision
                policy_decision = PolicyDecision(
                    decision=Decision.DEFENSE,
                    vfe_score=energy_for_decision,
                    threshold=2.0,
                    action_type=action_enum,
                    violated_axiom=violated_axiom,
                    reasoning=f"Decision Analysis:\n"
                              f"  Action Type: {action_enum.value}\n"
                              f"  VFE Score: {energy_for_decision:.4f}\n"
                              f"  Applied Threshold: 2.0000 (lowered: legitimacy_masking detected)\n"
                              f"  Result: DEFENSE\n\n"
                              f"  BLOCKED: Suspicious framing with elevated free energy."
                )
        else:
            policy_decision = self.decision_engine.decide(
                energy_for_decision,
                action_enum,
                violated_axiom
            )

        if self.verbose:
            print(f"Decision: {policy_decision.decision.value}")
            print(policy_decision.reasoning)

        return DEGDecision(
            safe=policy_decision.decision == Decision.COLLAB,
            energy_result=energy_result,
            extracted_intent=intent,
            reasoning=policy_decision.reasoning
        )

    @classmethod
    def from_config(cls, config_path: str) -> "DEGFramework":
        """
        Load DEG framework from a configuration file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Configured DEGFramework instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize components based on config
        extractor = cls._create_extractor(config.get("intent_extractor", {}))
        graph = cls._create_graph(config.get("axiomatic_graph", {}))
        calculator = cls._create_calculator(config.get("energy_calculator", {}))
        decision_eng = cls._create_decision_engine(config.get("decision_engine", {}))

        return cls(
            intent_extractor=extractor,
            axiomatic_graph=graph,
            energy_calculator=calculator,
            decision_engine=decision_eng,
            threshold=config.get("threshold", 0.7),
            verbose=config.get("verbose", False)
        )

    @staticmethod
    def _create_extractor(config: Dict[str, Any]) -> IntentExtractor:
        """Create intent extractor from config."""
        extractor_type = config.get("type", "nli")

        if extractor_type == "llm":
            return LLMIntentExtractor(
                model_name=config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
                device=config.get("device", "auto")
            )
        else:  # Default to NLIIntentExtractor (theoretically aligned)
            return NLIIntentExtractor(
                llm_model=config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
            )

    @staticmethod
    def _create_graph(config: Dict[str, Any]) -> AxiomaticGraph:
        """Create axiomatic graph from config."""
        graph_type = config.get("type", "local")

        if graph_type == "local" and "axioms_file" in config:
            graph = LocalAxiomaticGraph()
            graph.load_from_file(config["axioms_file"])
            return graph
        else:
            return LocalAxiomaticGraph()  # Default to local graph

    @staticmethod
    def _create_calculator(config: Dict[str, Any]) -> EnergyCalculator:
        """Create energy calculator from config."""
        # Dual-channel VP+RP calculator (RP via NLI meta-hypotheses, zero-training)

        # Resolve llr_stats_path relative to project root if not absolute
        llr_stats_path = config.get("llr_stats_path")
        if llr_stats_path and not os.path.isabs(llr_stats_path):
            # Resolve relative to project root (parent of deg)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            llr_stats_path = os.path.join(project_root, llr_stats_path)

        return NLIEnergyCalculator(
            threshold=config.get("threshold", 1.0),
            model_name=config.get("model_name", "cross-encoder/nli-deberta-v3-large"),
            use_corrected_formula=config.get("use_corrected_formula", True),
            nli_temperature=config.get("nli_temperature", 1.0),
            w_prior=config.get("w_prior", 1.0),
            sigma_nli=config.get("sigma_nli", 5.55),
            sigma_dr=config.get("sigma_dr", 4.34),
            llr_stats_path=llr_stats_path,
            llr_top_k=config.get("llr_top_k", 0),
        )

    @staticmethod
    def _create_decision_engine(config: Dict[str, Any]) -> DecisionEngine:
        """Create decision engine from config."""
        engine_type = config.get("type", "simple")

        if engine_type == "adaptive":
            return AdaptiveDecisionEngine(config)
        else:
            return SimpleDecisionEngine(config)


# Convenience factory functions

def create_full_deg(
    extractor_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    axioms_file: Optional[str] = None,
    threshold: float = 3.0,
    config: Optional[Dict[str, Any]] = None
) -> DEGFramework:
    """
    Create a DEG framework with full ML components.

    If a config dict is provided, uses it to configure all components
    including the AdaptiveDecisionEngine with action-dependent thresholds.
    Otherwise, falls back to parameter-based construction.

    Args:
        extractor_model: Model for intent extraction
        nli_model: NLI model for energy calculation
        axioms_file: Path to axioms JSON file (default: domain-specific axioms)
        threshold: Decision threshold (used when config is None)
        config: Optional full configuration dict (overrides other parameters)
    """
    import os

    # If config provided, use from_config for proper component setup
    if config is not None:
        return DEGFramework.from_config(config)

    # Default to domain-specific axioms if none provided
    if axioms_file is None:
        axioms_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "benchmark",
            "data",
            "phase1",
            "axioms_all_tagged.json"
        )

    graph = LocalAxiomaticGraph()

    # Load axioms if file exists
    if os.path.exists(axioms_file):
        graph.load_from_file(axioms_file)
        print(f"[DEG] Loaded {len(graph.axioms)} axioms from {axioms_file}")
    else:
        print(f"[DEG] Warning: Axioms file not found: {axioms_file}")
        print(f"[DEG] Using {len(graph.axioms)} default general axioms")

    # Recognition Prior (RP) is now computed via NLI meta-hypotheses (zero-training).
    # No separate DR model loading needed.

    return DEGFramework(
        intent_extractor=NLIIntentExtractor(llm_model=extractor_model),
        axiomatic_graph=graph,
        energy_calculator=NLIEnergyCalculator(
            threshold=threshold,
            model_name=nli_model,
            w_prior=1.0,
            sigma_nli=5.55,
            sigma_dr=4.34,
        ),
        decision_engine=AdaptiveDecisionEngine({
            "thresholds": {
                "WRITE": threshold,
                "READ": max(threshold * 0.833, 2.5),
                "default": threshold
            }
        }),
        threshold=threshold
    )


def create_deg_with_encoder(
    encoder_model_path: str = "benchmark/models/encoder/best_model.pt",
    nli_model: str = "cross-encoder/nli-deberta-v3-large",
    axioms_file: Optional[str] = None,
    threshold: float = 0.7,
    confidence_threshold: float = 1.0,
    device: str = "cuda"
) -> DEGFramework:
    """
    Create a DEG framework with trained encoder for intent extraction.

    This is MUCH faster than the NLI-based approach (~8ms vs 500ms per query).

    Args:
        encoder_model_path: Path to trained encoder checkpoint
        nli_model: NLI model for energy calculation
        axioms_file: Path to axioms JSON file (default: domain-specific axioms)
        threshold: Decision threshold for DEG
        confidence_threshold: Encoder confidence threshold
            - 1.0: Use encoder ONLY (fastest, ~8ms)
            - 0.7: Use encoder + LLM refinement (hybrid, ~100ms avg)
        device: Device to run on

    Returns:
        DEGFramework with encoder-based intent extraction
    """
    import os
    import torch

    # Load trained encoder
    print(f"[DEG] Loading encoder from: {encoder_model_path}")
    num_action_classes = max(ACTION_ONTOLOGY.values()) + 1
    encoder = IntentExtractorEncoder(
        model_name="microsoft/deberta-v3-small",
        num_actions=num_action_classes,
        num_patterns=len(SUSPICION_PATTERNS),
        max_span_length=20
    )

    checkpoint = torch.load(encoder_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"[DEG] Loaded encoder from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"[DEG] Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    else:
        encoder.load_state_dict(checkpoint)

    encoder.to(device)
    encoder.eval()

    # Create hybrid extractor
    hybrid_extractor = HybridIntentExtractor(
        encoder_model=encoder,
        llm_extractor_class=NLIIntentExtractor,
        confidence_threshold=confidence_threshold
    )

    # Preload encoder
    hybrid_extractor.preload()

    print(f"[DEG] Hybrid intent extractor created (threshold={confidence_threshold})")
    if confidence_threshold >= 1.0:
        print(f"[DEG] Mode: Encoder-only (fastest, ~8ms per query)")
    else:
        print(f"[DEG] Mode: Hybrid with LLM refinement (~100ms avg)")

    # Load axioms
    if axioms_file is None:
        axioms_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "benchmark",
            "data",
            "phase1",
            "axioms_all_tagged.json"
        )

    graph = LocalAxiomaticGraph()

    # Load axioms if file exists
    if os.path.exists(axioms_file):
        graph.load_from_file(axioms_file)
        print(f"[DEG] Loaded {len(graph.axioms)} axioms from {axioms_file}")
    else:
        print(f"[DEG] Warning: Axioms file not found: {axioms_file}")
        print(f"[DEG] Using {len(graph.axioms)} default general axioms")

    # Create action-type-aware decision engine
    # READ operations get lenient threshold (1.5) - balanced approach
    # WRITE operations get strict threshold (1.0)
    from deg.decision_engine import AdaptiveDecisionEngine, Action

    # Use the threshold parameter to scale decision thresholds
    # Base scale: threshold=1.0 means WRITE=1.0, READ=1.5
    # Higher threshold values scale both thresholds proportionally
    base_write_threshold = 1.0
    base_read_threshold = 1.5
    scale_factor = threshold / base_write_threshold

    decision_engine = AdaptiveDecisionEngine({
        "thresholds": {
            "WRITE": base_write_threshold * scale_factor,
            "READ": base_read_threshold * scale_factor,
            "default": base_write_threshold * scale_factor
        },
        "lambda_epistemic": 2.0
    })

    return DEGFramework(
        intent_extractor=hybrid_extractor,
        axiomatic_graph=graph,
        energy_calculator=NLIEnergyCalculator(
            threshold=threshold,  # Energy calculator still uses base threshold
            model_name=nli_model
        ),
        decision_engine=decision_engine,  # Use adaptive decision engine
        threshold=threshold  # Legacy threshold (kept for compatibility)
    )
